import unittest
import os 
import shutil

from tfile import TFileService
import PhysicsTools.HeppyCore.framework.config as cfg

class ServiceTestCase(unittest.TestCase):

    def test_tfile(self):
        config = cfg.Service(TFileService, 
                             'myhists', 
                             fname = 'histos.root', 
                             option = 'recreate')
        dummy = None
        dirname = 'test_dir'
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)
        fileservice = TFileService(config, dummy, dirname)
        fileservice.start()
        fileservice.stop()
        shutil.rmtree(dirname)

if __name__ == '__main__':
    unittest.main()
