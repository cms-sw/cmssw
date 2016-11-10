from dataset import * 
import eostools

import unittest 
import os 
import shutil

BASE_DIR = 'datasets'

def create_dataset(name, number_of_files, basedir=BASE_DIR):
    if not os.path.isdir(basedir):
        os.mkdir(basedir)
    old = os.getcwd()
    os.chdir(basedir)
    if os.path.isdir(name):
        shutil.rmtree(name)
    os.mkdir(name)
    os.chdir(name)
    for i in range(number_of_files):
        os.system('touch file_{i:d}.root'.format(i=i))
    os.chdir(old)

class TestDataset(unittest.TestCase):

    def test_local(self):
        n_files = 10
        create_dataset('ds1', n_files)
        ds1 = LocalDataset('/ds1', 'datasets', '.*root')
        self.assertEqual( len(ds1.listOfGoodFiles()), n_files)
        # shutil.rmtree('datasets')

    def test_eos(self):
        cbern = '/eos/cms/store/cmst3/user/cbern'
        if not 'CMSSW_BASE' in os.environ:
            return
        if not eostools.fileExists(cbern): 
            return 
        ds1 = EOSDataset('/EOSTests/ds1', 
                         cbern,
                         '.*root') 
        self.assertEqual(len(ds1.listOfGoodFiles()), 10)

    def test_eos_fail(self):
        if not 'CMSSW_BASE' in os.environ:
            return
        self.assertRaises( ValueError, 
                           EOSDataset, 
                           'not_existing_basedir',
                           'not_exiting_dataset',
                           '.*root')
        # should test that we fail when a plain file is provided 
        # instead of a directory.. but eostools not set up for that yet.


if __name__ == '__main__':
    unittest.main()
