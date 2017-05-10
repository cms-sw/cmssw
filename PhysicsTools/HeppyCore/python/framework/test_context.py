import unittest
import os
import copy

from context import get_name

class TestContext(unittest.TestCase): 
    
    def test_cms(self):
        env = dict(CMSSW_BASE='.')
        self.assertEqual(get_name(env), 'cms')

    def test_fcc(self):
        env = dict(PODIO='.', FCCEDM='.', FCCPHYSICS='.', HEPPY='.')        
        self.assertEqual(get_name(env), 'fcc')

    def test_fcc_cms(self):
        env = dict(PODIO='.', FCCEDM='.', FCCPHYSICS='.', HEPPY='.',
                   CMSSW_BASE='.')        
        self.assertRaises( ValueError, get_name, env )
        

if __name__ == '__main__':
    unittest.main()
