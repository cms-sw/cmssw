import unittest
from ROOT import TFile
from tree import Tree

class TreeTestCase(unittest.TestCase):

    def test_fill(self):
        fi = TFile('tree.root','RECREATE')
        tr = Tree('test_tree', 'A test tree')
        tr.var('a')
        tr.var('b')
        tr.fill('a', 3)
        tr.fill('a', 4)
        tr.fill('b', 5)
        tr.tree.Fill()
        fi.Write()
        fi.Close()

    def test_read(self):
        fi = TFile('tree.root')
        tr = fi.Get('test_tree')
        self.assertEqual(tr.GetEntries(), 1)
        tr.GetEntry(0)
        self.assertEqual(tr.a, 4)

    def test_iterate(self):
        fi = TFile('tree.root')
        tr = fi.Get('test_tree')
        for ev in tr:
            self.assertEqual(ev.a, 4)
            break

    def test_cwn(self):
        fi = TFile('tree2.root','RECREATE')
        tr = Tree('test_tree', 'A test tree')
        tr.var('nvals', the_type=int)
        tr.vector('x', 'nvals', 20)
        tr.fill('nvals', 10)
        tr.vfill('x', range(10))
        tr.tree.Fill()
        tr.reset()
        tr.fill('nvals', 5)
        tr.vfill('x', range(5))
        tr.tree.Fill()        
        fi.Write()
        fi.Close()
        

if __name__ == '__main__':
    unittest.main()
