import unittest
from floodfill import FloodFill

class Node(int):
    def __init__(self, *args):
        self.linked = []
        self.block_label = None
        super(Node, self).__init__(*args)

    def accept(self, visitor):
        notseen = visitor.visit(self)
        if notseen:
            for elem in self.linked:
                elem.accept(visitor)

    def __str__(self):
        return super(Node, self).__str__() + str(self.linked)

class Graph(object):
    def __init__(self, edges):
        self.nodes = dict()
        for e1, e2 in edges:
            node1, node2 = None, None
            if e1:
                node1 = self.nodes.get(e1, False)
                if not node1:
                    node1 = Node(e1)
                    self.nodes[e1] = node1
            if e2:
                node2 = self.nodes.get(e2, False)
                if not node2:
                    node2 = Node(e2)
                    self.nodes[e2] = node2
            if node1 and node2:  
                node1.linked.append(node2)
                node2.linked.append(node1)

        
class TestFloodFill(unittest.TestCase):
    def test_1(self):
        graph = Graph( [ (1,2), (1,3), (4,None) ] )
        floodfill = FloodFill(graph.nodes.values())
        self.assertEqual(floodfill.groups.keys(), [0,1])
        self.assertEqual(floodfill.groups.values()[0], [1,2,3])
        self.assertEqual(floodfill.groups.values()[1], [4])
        
    def test_2(self):
        graph = Graph( [ (1,2), (2,3), (3,4), (5, 6) ] )
        floodfill = FloodFill(graph.nodes.values())
        self.assertEqual(floodfill.groups.keys(), [0,1])
        self.assertEqual(floodfill.groups.values()[0], [1,2,3,4])
        self.assertEqual(floodfill.groups.values()[1], [5,6])

    def test_regroup(self):
        graph = Graph( [ (1,2), (2,3), (3,4), (5, 6) ] )
        floodfill = FloodFill(graph.nodes.values())
        self.assertEqual(floodfill.groups.keys(), [0,1])
        graph.nodes[1].linked.remove(graph.nodes[2])
        graph.nodes[2].linked.remove(graph.nodes[1])
        floodfill = FloodFill(floodfill.groups[0],
                              first_label=floodfill.label)
        self.assertEqual(floodfill.groups.keys(), [2,3])
        self.assertEqual(floodfill.groups[2], [1])
        self.assertEqual(floodfill.groups[3], [2,3,4])
        

    
if __name__ == '__main__':
    unittest.main()
