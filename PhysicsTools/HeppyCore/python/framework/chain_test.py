import unittest
import os
import shutil

from chain import Chain
from PhysicsTools.HeppyCore.utils.testtree import create_tree

testfname = 'test_tree.root'

class ChainTestCase(unittest.TestCase):

    def setUp(self):
        self.chain = Chain(testfname, 'test_tree')

    def test_file(self):
        '''Test that the test file exists'''
        self.assertTrue(os.path.isfile(testfname))

    def test_wrong_filename(self):
        self.assertRaises(ValueError,
                          Chain, 'non_existing_file.root')

    def test_guess_treename(self):
        chain = Chain(testfname)
        self.assertEqual(len(self.chain), 100)        

    def test_load_1(self):
        '''Test that the chain has the correct number of entries'''
        self.assertEqual(len(self.chain), 100)

    def test_load_2(self):
        '''Test chaining of two files.'''
        tmpfile = testfname.replace('test_tree', 'test_tree_2_tmp')
        shutil.copyfile(testfname, tmpfile)
        chain = Chain(testfname.replace('.root', '*.root'), 'test_tree')
        self.assertEqual(len(chain), 200)
        os.remove(tmpfile)
    
    def test_load_3(self):
        '''Test LFN/root-fn loading'''
        chain = Chain(["root://{0}".format(os.path.abspath(testfname))], 'test_tree')
        self.assertEqual(len(chain), 100)

    def test_iterate(self):
        '''Test iteration'''
        for ev in self.chain:
            pass
        self.assertTrue(True)

    def test_get(self):
        '''Test direct event access'''
        event = self.chain[2]
        self.assertEqual(event.var1, 2.)


if __name__ == '__main__':
    create_tree(testfname)
    unittest.main()
