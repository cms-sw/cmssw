from DAG import Node,  BreadthFirstSearchIterative, BreadthFirstSearch
import unittest 

class TreeTestCase( unittest.TestCase ):

    def setUp(self):
        '''
        called before every test. 
        0 and 8 are root/head nodes
    
    
        8
         \
          \
           9
            \
             \
              4  
             /  
            /  
           1--5--7
          / \
         /   \
        0--2  6
         \   /
          \ /
           3

    '''
    # building all nodes
        self.nodes = dict( (i, Node(i) ) for i in range(10) )
    
        self.nodes[0].add_child(self.nodes[1])
        self.nodes[0].add_child(self.nodes[2])
        self.nodes[0].add_child(self.nodes[3])
        self.nodes[1].add_child(self.nodes[4])
        self.nodes[1].add_child(self.nodes[5])
        self.nodes[1].add_child(self.nodes[6])
        self.nodes[5].add_child(self.nodes[7])
        self.nodes[8].add_child(self.nodes[9])
        self.nodes[9].add_child(self.nodes[4])
        self.nodes[3].add_child(self.nodes[6])
    
               
        
    def test_BFS_visitor_pattern_iterative_undirected(self):
        BFS = BreadthFirstSearchIterative(self.nodes[0],"undirected")
        # the result is equal to [0, 1, 2, 3, 4, 5, 6, 9, 7, 8]
        values=[]
        for x in BFS.result:
            values.append(x.value)        
        self.assertEqual(values, [0, 1, 2, 3, 4, 5, 6, 9, 7, 8] )
        
    def test_BFS_visitor_pattern_children(self):
        BFS = BreadthFirstSearch(self.nodes[0],"children")
          # the result is equal to [0, 1, 2, 3, 4, 5, 6, 7]
        values=[]
        for x in BFS.result:
            values.append(x.value)
        self.assertEqual(values, range(8) )

    def test_BFS_visitor_pattern_undirected(self):
            
        BFS = BreadthFirstSearch(self.nodes[0],"undirected")
        # the result is equal to [0, 1, 2, 3, 4, 5, 6, 9, 7, 8]
        values=[]
        for x in BFS.result:
            values.append(x.value)        
        self.assertEqual(values, [0, 1, 2, 3, 4, 5, 6, 9, 7, 8] )
    
   
if __name__ == '__main__':
    unittest.main()
