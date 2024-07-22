from collections import deque 
import heapq
import sys

# Problem 1. You are expected to handle AVL trees of a maximum of n nodes. You decide to implement them using arrays.
#  1/ What is the size of the array you should allocate to make sure you will have enough space to store your AVL tree in the array? 
#  2/ Implement insertion and deletion methods on an AVL tree as an array.
class AVLTree:
  def __init__(self, max_nodes):
    self.max_nodes = max_nodes
    self.tree = [None] * max_nodes

  def insert(self, value):
    self._insert_recursive(value, 0)

  def _insert_recursive(self, value, index):
    # recursively insert a val into tree
    if self.tree[index] is None:
      # if current node is empty, insert val
      self.tree[index] = value
      return

    if value < self.tree[index]:
      # if val is less than current node's val, traverse left
      self._insert_recursive(value, 2 * index + 1)
    else:
      # if val is greater than or equal to current node's val, traverse right
      self._insert_recursive(value, 2 * index + 2)

  def delete(self, value):
    self.deleting(value, 0)

  def deleting(self, value, index):
    # doesn't exist or out of bounds
    if index >= self.max_nodes or self.tree[index] is None:
      return

    # if val to be deleted is less than val at the current node,
    # recursively delete in left subtree
    if value < self.tree[index]:
      if 2 * index + 1 < self.max_nodes:
        self.deleting(value, 2 * index + 1) 

    # if val to be deleted is greater than val at the current node,
    # recursively delete in right subtree
    elif value > self.tree[index]:
        if 2 * index + 2 < self.max_nodes:
          self.deleting(value, 2 * index + 2)
    
    # if val to be deleted matches val at current node
    else:
      # if left child is None
      if self.tree[2 * index + 1] is None:
        # if right child is also None, it's a leaf node, set to None
        if self.tree[2 * index + 2] is None:
          self.tree[index] = None 
        else:
          # find the min val in the right subtree (successor)
          min_value_index = self._find_min_index(2 * index + 2)
          # replace current node with successor
          self.tree[index] = self.tree[min_value_index]
          # delete successor node
          self.tree[min_value_index] = None
      else:
        # if left child is not None, find the max val in left subtree (predecessor)
        max_value_index = self._find_max_index(2 * index + 1)
        # replace current node with predecessor
        self.tree[index] = self.tree[max_value_index]
        # delete predecessor node
        self.tree[max_value_index] = None


  def _find_min_index(self, index):
    # find min val index starting from given index
    while 2 * index + 1 < self.max_nodes and self.tree[2 * index + 1] is not None:
      index = 2 * index + 1
    return index

  def _find_max_index(self, index):
    # find max val index starting from given index
    while 2 * index + 2 < self.max_nodes and self.tree[2 * index + 2] is not None:
      index = 2 * index + 2
    return index




# Problem 2. Given a graph G, implement
#   1/ Depth-First Search (DFS)
def dfs(graph, start, visited=None):
  if visited is None:
    visited = set()
    
  visited.add(start)
  # print node when visited
  print(start, end=' ')
    
  # visit all neighbors
  for neighbor in graph[start]:
    # if neighbor hasn't been visited, recursively call DFS on it
    if neighbor not in visited:
      dfs(graph, neighbor, visited)
    
  return visited

# 2/ Breadth-First Search (BFS), to search for a given element in G.
def bfs(graph, start):  
  visited = set()  
  queue = deque([start])
  visited_order = [] 

  # loop until queue is empty
  while queue:  
    vertex = queue.popleft()  
    if vertex not in visited:
      # record the visited node  
      visited_order.append(vertex)
      visited.add(vertex)  
      # extend the queue only with unvisited neighbors of current node
      queue.extend([v for v in graph[vertex] if v not in visited])
  return visited_order  
 



# Problem 3. Given a weighted directed graph G and two nodes in G (denotes s and t), implement a method that determines the minimum cost to travel from s to t.  
def dijkstra(graph, s, t):
  # set distances to infinity for all nodes
  distances = {node: float('inf') for node in graph}
  # set distance from start node to itself as 0
  distances[s] = 0
  # priority queue to store nodes based on their tentative distances
  pq = [(0, s)]

  while pq:
    # pop node with smallest distance from priority queue
    current_distance, current_node = heapq.heappop(pq)
        
    # if distance along current path is greater than the shortest known distance no need to continue exploring it
    if current_distance > distances[current_node]:
      continue
        
    # iterate through neighbors of current node
    for neighbor, weight in graph[current_node].items():
       # add total distance to reach neighbor
        total_distance = current_distance + weight

        # if new distance is smaller than existing distance, fix it
        if total_distance < distances[neighbor]:
          distances[neighbor] = total_distance
          # add neighbor to the pq with new distance
          heapq.heappush(pq, (total_distance, neighbor))
  # return min distance from s to t
  return distances[t]        



# Problem 4. Given a weighted graph G, implement a method that determines which edges of G form part of its minimum spanning tree T.
class Graph:
  def __init__(self, vertices):
    self.V = vertices
    self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

  # finds the vertex with minimum key value not yet included in the MST
  def minKey(self, key, mstSet):
    min = sys.maxsize
    min_index = -1

    for v in range(self.V):
      # check if vertex is not in MST and has a smaller key value
      if key[v] < min and mstSet[v] == False:
        min = key[v]
        min_index = v
    # return index of vertex with min key
    return min_index

  def primMST(self):
    key = [sys.maxsize] * self.V
    parent = [None] * self.V
    key[0] = 0 # start the MST with the 1st vertex by setting its key to 0
    mstSet = [False] * self.V
    mst = []

    parent[0] = -1 # root of MST has no parent

    for _ in range(self.V):
      u = self.minKey(key, mstSet)
      # mark this vertex as included in MST
      mstSet[u] = True
      # loop through all vertices to update key and parent values
      for v in range(self.V):
        if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
          key[v] = self.graph[u][v]  # update key with smaller edge weight
          parent[v] = u
    # construct MST by adding edges (parent, child) to list
    for i in range(1, self.V):
      mst.append((parent[i], i))
      
    return mst



# Problem 5. Implement a method that receives and sorts “almost sorted arrays” = arrays in which elements
# are either in order or not farther than k spots away from where they should be. What’s the time complexity of your approach? Can you do better?
def almost_sorted(arr, k):
  # empty list to store sorted elements
    sorted_arr = []
    
    # min-heap with the first k+1 elements of array
    min_heap = arr[:k+1]
    heapq.heapify(min_heap)
    
    # iterate through remaining elements of array
    for i in range(k+1, len(arr)):
      # add min element from min-heap and append it to sorted array
      min_element = heapq.heappop(min_heap)
      sorted_arr.append(min_element)
        
      # add the next element from array to min-heap
      heapq.heappush(min_heap, arr[i])
    
    # add and append remaining elements from min-heap to sorted array
    while min_heap:
      min_element = heapq.heappop(min_heap)
      sorted_arr.append(min_element)
    
    return sorted_arr




# Problem 6. You are tasked with implementing a system that simulates traffic at a traffic light.
# What data structure do you use and why? Elaborate on the time and space complexity of your approach to motivate your choice.
def traffic_light(line, green_light):
  queue = deque(line) 

  if green_light < len(queue):
    # remove the specified number of cars from the queue
    for _ in range(green_light):
      queue.popleft() 
  else:
    # if green light duration is longer than the queue, empty the whole queue
    queue.clear()

  # return remaining cars in the queue
  return list(queue)



# --------------------------------------Testing---------------------------------------------------

# Problem 1
print("Testing Problem 1")
print(" The size of the array you should allocate to make sure you will have enough space to store your AVL tree in the array is: 2^n -1 where n is the number of nodes in the AVL tree. ")

# testing insertion method
avl = AVLTree(10)
avl.insert(10)
avl.insert(5)
avl.insert(15)
avl.insert(3)
avl.insert(7)
avl.insert(12)
avl.insert(20)
print("\n Testing insertion method  \n Expected: [10, 5, 15, 3, 7, 12, 20, None, None, None]  \n Actual:  ", avl.tree)

# testing deletion method
avl.delete(5)
avl.delete(15)
print("\n Testing deletion method  \n Expected: [10, 3, 12, None, 7, None, 20, None, None, None]  \n Actual:  ", avl.tree)


# Problem 2
print("\nTesting Problem 2")
graph = { 
  'A': ['B', 'C'],  
  'B': ['D', 'E'],  
  'C': ['F'],  
  'D': [],  
  'E': ['F'],  
  'F': [] 
} 

# testing DFS method
print(" Testing DFS method \n Expected: A B D E F C \n Actual:   ", end='')
result = dfs(graph, 'A')

# testing BFS method
result = bfs(graph, 'A')
print("\n\n Testing BFS method \n Expected: A B C D E F \n Actual:  ", ' '.join(result))

# Problem 3
print("\nTesting Problem 3")
g = {
    's': {'A': 1, 'B': 2},
    'A': {'B': 1, 'C': 1},
    'B': {'C': 1},
    'C': {'t': 1},
    't': {}
    }

print(" Testing Dijkstra method \n Expected: 3 \n Actual:  ", dijkstra(g, 's', 't'))


# Problem 4
print("\nTesting Problem 4") 
g = Graph(5)
g.graph = [[0, 2, 0, 6, 0],
           [2, 0, 3, 8, 5],
           [0, 3, 0, 0, 7],
           [6, 8, 0, 0, 9],
           [0, 5, 7, 9, 0]]

mst = g.primMST()
print(" Testing Prims method \n Expected: [(0, 1), (1, 2), (0, 3), (1, 4)] \n Actual:  ", mst)

# Problem 5
print("\nTesting Problem 5")
print(" The time complexity of this approach is: O(n log(k)). If k is very large compared to n, you can achieve better performance by using other sorting algorithms which have an average time complexity of O(n log(n)).")
arr = [3, 1, 2, 5, 4, 6]
k = 2
print("\nTesting Almost sorted method \n Expected: [1, 2, 3, 4, 5, 6] \n Actual:  ", almost_sorted(arr, k))


# Problem 6
print("\nTesting Problem 6")
line = ["Tacoma", "Camry", "Tesla", "Jeep"]
print(" I used a queue, because it follows a First-In, First-Out (FIFO) Order. The time complexity of my approach is O(n) and the space complexity is O(n).")
print(" Testing Traffic lights method \n Expected: ['Tesla', 'Jeep'] \n Actual:  ", traffic_light(line, 2))
print(" Expected: [] \n Actual:  ", traffic_light(line, 8))