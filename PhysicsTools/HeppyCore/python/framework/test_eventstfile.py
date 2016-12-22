import unittest

from ROOT import TFile

from eventstfile import Events
from PhysicsTools.HeppyCore.utils.testtree import create_tree

testfname = 'test_tree.root'

class EventsTFileTestCase(unittest.TestCase):

    def setUp(self):
        self.fname = create_tree()
        rootfile = TFile(self.fname)
        self.events = Events(testfname, 'test_tree')

    def test(self):
        event = self.events.to(2)
        for iev, ev in enumerate(self.events):
            self.assertEqual(iev, ev.var1)
 
if __name__ == '__main__':
    unittest.main()
