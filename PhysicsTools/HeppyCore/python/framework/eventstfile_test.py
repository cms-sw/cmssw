import unittest

from eventstfile import Events
from PhysicsTools.HeppyCore.utils.testtree import create_tree

testfname = 'test_tree.root'

class EventsTFileTestCase(unittest.TestCase):

    def test(self):
        events = Events(testfname, 'test_tree')
        event = events.to(2)
        for iev, ev in enumerate(events):
            self.assertEqual(iev, ev.var1)
 
if __name__ == '__main__':
    create_tree(testfname)
    unittest.main()
