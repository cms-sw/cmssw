import pprint

from collections import deque

'''Directed Acyclic Graph (DAG) with floodfill and breadth first traversal algorithms

Each node may have several children. 
Each node may have several parents. The DAG may have multiple roots ( a node without a parent)
It has no loops when directed, but may have loops when traversed in an undirected way

Traversals
1: deal with all nodes at the same level and then with all children of these nodes etc (breadth-first search or BFS)
2: deal with all children and finally with the node (depth-first search of DFS)  

A "visitor pattern" is used to allow the algorithms to be separated from the object 
on which it operates and without modifying the objects structures (eg a visited flag can be 
owned by the algorithm)

The visitor pattern also allows the visit method to dynamically depend on both the object and the visitor

example of setting up Nodes:
        self.nodes = dict( (i, Node(i) ) for i in range(10) 
        self.nodes[0].add_child(self.nodes[1])
        self.nodes[0].add_child(self.nodes[2])
traversing nodes:        
        BFS = BreadthFirstSearchIterative(self.nodes[0],"undirected")
        see alos test_DAG.py
'''


class Node(object):
    '''
    Implements a Directed Acyclic Graph: 
    each node has an arbitrary number of children and parents
    There are no loops in the directed DAG
    But there may be loops in the undirected version of the DAG

    attributes:
       value = the item of interest (around which the node is wrapped)
       children = list of child nodes
       parents  = list of parent node
       undirected_links = combined list of parents and children
    '''

    def __init__(self, value):
        '''constructor. 
        value can be anything, even a complex object. 
        example:
           newnode=Node(uniqueid)
        '''
        self.value = value   # wrapped object
        self.children = []
        self.parents = []
        self.undirected_links = [] #the union of the parents and children (other implementations possible)

    def get_value(self):
        return self.value

    def accept(self, visitor):
        visitor.visit(self)

    def add_child(self, child):
        '''set the children'''
        self.children.append(child)
        child.add_parent(self)
        self.undirected_links.append(child)

    def add_parent(self, parent):
        '''set the parents'''
        self.parents.append(parent)
        self.undirected_links.append(parent)

    def remove_all_links_to(self,toremove):
        '''checks for element toremove in the list of children and parents and
           removes any links from both this and from the toremove node
        '''
        if (toremove in self.parents):
            self.parents.remove(toremove)
            toremove.children.remove(self)
        if (toremove in self.children):
            self.children.remove(toremove)
            toremove.parents.remove(self)        

    def get_linked_nodes(self, type):  #ask colin, I imagine there is a more elegant Python way to do this
                                        #alice todo make type a enumeration and not a string?
        '''return a list of the linked children/parents/undirected links'''
        if (type is "children"):
            return self.children
        if(type is "parents"):
            return self.parents
        if(type is "undirected"):
            return self.undirected_links

    def __repr__(self):
        '''unique string representation'''
        return self.__str__()

    def __str__(self):
        '''unique string representation'''         
        return str('node: {val} {children}'.format(
            val = self.value,
            children = self.children
        ) )    


class BreadthFirstSearch(object):

    def __init__(self,root, link_type):
        '''Perform the breadth first recursive search of the nodes'''
        self.result = []
        self.root = root
        self.visited = dict()
        self.bfs_recursive([root],link_type)

    def visit(self, node):
        if self.visited.get(node, False):
            return
        self.result.append( node )
        self.visited[node] = True

    def bfs_recursive(self,nodes, link_type ):
        '''Breadth first recursive implementation
        each recursion is one level down the tree
        link_type can be "children", "parents","undirected" '''
        link_nodes = []
        if len(nodes) is 0:
            return 

        for node in nodes: # collect a list of all the next level of nodes
            if (self.visited.get(node, False)):  
                continue              
            link_nodes.extend(node.get_linked_nodes(link_type))        
        for node in nodes: #add these nodes onto list and mark as visited
            if (self.visited.get(node, False)):  
                continue            
            node.accept(self)

        self.bfs_recursive(link_nodes,  link_type)


class BreadthFirstSearchIterative(object):

    def __init__(self,root, link_type):
        '''Perform the breadth first iterative search of the nodes'''
        self.visited = {}
        self.result = []
        self.bfs_iterative(root,link_type)       

    def visit(self, node):
        if self.visited.get(node, False):
            return
        self.result.append( node )
        self.visited[node] = True

    def bfs_iterative(self,node, link_type ):
        '''Breadth first iterative implementation
        using a deque to order the nodes 
        link_type can be "children", "parents","undirected" '''

        # Create a deque for the Breadth First Search
        todo = deque()
        todo.append( node)

        while len(todo):
            node = todo.popleft()
            if self.visited.get(node,False): #check if already processed
                continue
            node.accept(self)
            for linknode in node.get_linked_nodes(link_type):
                if self.visited.get(linknode,False): #check if already processed
                    continue
                todo.append( linknode)  


class DAGFloodFill(object):

    def __init__(self, elements, first_label = 1):
        '''Iterate through all nodes and 
        use Breadth first search to find connected groups'''
        self.visited = {}
        self.label = first_label
        self.visited = dict()
        self.blocks = []
        for uid, node in elements.iteritems():
            if self.visited.get(node, False): #already done so skip the rest
                continue

            #find connected nodes
            bfs = BreadthFirstSearchIterative(node,"undirected")

            # set all connected elements to have a visited flag =true
            for n in bfs.result :
                self.visited.update({n: True})
            #add into the set of blocks
            self.blocks.append( bfs.result)
            self.label += 1



