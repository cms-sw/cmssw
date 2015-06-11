from dataset import * 

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
    

if __name__ == '__main__':
    import os
    import sys 
    if not os.environ.get('CMSSW_BASE', False):
        sys.exit(1)
    unittest.main()
