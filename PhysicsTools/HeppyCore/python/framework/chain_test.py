import unittest
import os
import shutil

from PhysicsTools.HeppyCore.framework.chain import Chain

class ChainTestCase(unittest.TestCase):

    def setUp(self):
        self.file = '../../test/test_tree.root'
        self.chain = Chain(self.file, 'test_tree')

    def test_file(self):
        self.assertTrue(os.path.isfile(self.file))

    def test_guess_name(self):
        self.assertRaises(ValueError,
                          Chain, 'self.file')

    def test_load_1(self):
        self.assertEqual(len(self.chain), 100)

    def test_load_2(self):
        tmpfile = self.file.replace('test_tree', 'test_tree_2_tmp')
        shutil.copyfile(self.file, tmpfile)
        chain = Chain(self.file.replace('.root', '*.root'), 'test_tree')
        self.assertEqual(len(chain), 200)
        os.remove(tmpfile)

    def test_iterate(self):
        for ev in self.chain:
            pass
        self.assertTrue(True)

    def test_get(self):
        event = self.chain[2]
        self.assertEqual(event.var1, 2.)


if __name__ == '__main__':
    unittest.main()
