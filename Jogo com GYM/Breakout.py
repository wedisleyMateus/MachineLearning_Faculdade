# import gym
# import numpy as np
#
# env = gym.make("Breakout-v4", render_mode="human")
#
# q_table = {}
# alpha = 0.1
# gamma = 0.99
# epsilon = 1.0
# epsilon_decay = 0.995
# epsilon_min = 0.1
#
# def choose_action(state):
#     if np.random.random() < epsilon:
#         return env.action_space.sample()
#     else:
#         return np.argmax(q_table.get(state, np.zeros(env.action_space.n)))
#
# def update_q_table(state, action, reward, next_state):
#     current_q = q_table.get(state, np.zeros(env.action_space.n))[action]
#     max_future_q = np.max(q_table.get(next_state, np.zeros(env.action_space.n)))
#     new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
#     q_table.setdefault(state, np.zeros(env.action_space.n))[action] = new_q
#
# for episode in range(500):
#     state, info = env.reset()
#     state = tuple(state.flatten())
#     total_reward = 0
#
#     for _ in range(300):
#         action = choose_action(state)
#         next_state, reward, done, truncated, info = env.step(action)
#         next_state = tuple(next_state.flatten())
#         update_q_table(state, action, reward, next_state)
#
#         state = next_state
#         total_reward += reward
#
#         if done:
#             break
#
#     epsilon = max(epsilon * epsilon_decay, epsilon_min)
#
#     print(f"Episódio: {episode}, Recompensa Total: {total_reward}, Epsilon: {epsilon}")
#
# env.close()


import gym
import numpy as np
from collections import defaultdict

env = gym.make("Breakout-v4", render_mode="human")

q_table = defaultdict(lambda: np.zeros(env.action_space.n))
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

def preprocess_state(state):

    gray_state = np.mean(state, axis=-1)

    downsampled_state = gray_state[::2, ::2]

    discretized_state = (downsampled_state // 32).astype(int)

    return tuple(discretized_state.flatten())

def choose_action(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

def update_q_table(state, action, reward, next_state):
    current_q = q_table[state][action]
    max_future_q = np.max(q_table[next_state])
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
    q_table[state][action] = new_q

for episode in range(500):
    state, info = env.reset()
    state = preprocess_state(state)
    total_reward = 0

    for _ in range(1000):
        action = choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = preprocess_state(next_state)
        update_q_table(state, action, reward, next_state)

        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    print(f"Episódio: {episode}, Recompensa Total: {total_reward}, Epsilon: {epsilon}")

env.close()
