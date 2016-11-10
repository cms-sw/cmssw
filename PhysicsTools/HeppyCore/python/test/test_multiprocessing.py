import unittest
import shutil
import tempfile
import os
import subprocess
import copy
import glob
from simple_example_cfg import config, stopper 
from PhysicsTools.HeppyCore.utils.testtree import create_tree, remove_tree
from PhysicsTools.HeppyCore.framework.looper import Looper
from PhysicsTools.HeppyCore.framework.exceptions import UserStop
import PhysicsTools.HeppyCore.framework.context as context
from ROOT import TFile

import logging
logging.getLogger().setLevel(logging.ERROR)

class Options(object): 
    pass

class TestMultiProcessing(unittest.TestCase):

    def setUp(self):
        self.fname = create_tree()
        self.fname2 = self.fname.replace('.root','_2.root')
        shutil.copy(self.fname, self.fname2)
        rootfile = TFile(self.fname)
        self.nevents = rootfile.Get('test_tree').GetEntries()
        self.outdir = tempfile.mkdtemp()
        logging.disable(logging.CRITICAL)
        
    def tearDown(self):
        shutil.rmtree(self.outdir)
        logging.disable(logging.NOTSET)
        os.remove(self.fname2)

    def test_multiprocessing(self): 
        from PhysicsTools.HeppyCore.framework.heppy_loop import create_parser, main
        parser = create_parser()
        options, args = parser.parse_args()
        options.iEvent = None
        options.nprint = 0
        cfg = '/'.join( [ context.heppy_path, 
                          'test/simple_multi_example_cfg.py' ] )
        main(options, [self.outdir, cfg], parser)
        wcard = '/'.join([self.outdir, 
                          'test_component_Chunk*',
                          'PhysicsTools.HeppyCore.analyzers.examples.simple.SimpleTreeProducer.SimpleTreeProducer_tree/simple_tree.root'
                          ])
        output_root_files = glob.glob(wcard)
        self.assertEqual(len(output_root_files),2)
                
    # def test_heppy_batch(self):
    #     cmd = ['heppy_batch.py',
    #            '-o',
    #            '{}'.format(self.outdir), 
    #            '-b',
    #            'nohup ./batchScript.sh &', 
    #            'simple_multi_example_cfg.py']
    #     FNULL = open(os.devnull,'w')
    #     p = subprocess.Popen(cmd, stdout=FNULL, 
    #                          stderr=subprocess.STDOUT)
    #     # p.communicate()
    #     p.wait()
    #     import time 
    #     wcard = '/'.join([self.outdir, 
    #                       'test_component_Chunk*',
    #                       'PhysicsTools.HeppyCore.analyzers.examples.simple.SimpleTreeProducer.SimpleTreeProducer_tree/simple_tree.root'
    #                       ])
    #     output_root_files = []
    #     print wcard
    #     for i in range(50): 
    #         # waiting for max 10 seconds for the nohup processes
    #         # to complete and the files to appear.
    #         print 'wait'
    #         time.sleep(1)
    #         output_root_files = glob.glob(wcard)
    #         if len(output_root_files):
    #             break 
    #     self.assertEqual(len(output_root_files),2)
        

if __name__ == '__main__':

    unittest.main()
