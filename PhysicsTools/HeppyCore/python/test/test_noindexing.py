import unittest
import shutil
import tempfile
import os 
from simple_example_noindexing_cfg import config
from PhysicsTools.HeppyCore.utils.testtree import create_tree, remove_tree
from PhysicsTools.HeppyCore.framework.looper import Looper
from ROOT import TFile

import logging
logging.getLogger().setLevel(logging.ERROR)

class TestNoIndexing(unittest.TestCase):

    def setUp(self):
        self.fname = create_tree()
        rootfile = TFile(self.fname)
        self.nevents = rootfile.Get('test_tree').GetEntries()
        self.outdir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.outdir)    

    def test_all_events_processed(self):
        loop = Looper( self.outdir, config,
                       nEvents=None,
                       nPrint=0,
                       timeReport=True)
        loop.loop()
        loop.write()
        logfile = open('/'.join([self.outdir, 'log.txt']))
        nev_processed = None
        for line in logfile:
            if line.startswith('number of events processed:'):
                nev_processed = int(line.split(':')[1])
        logfile.close()
        self.assertEqual(nev_processed, self.nevents)
        # checking the looper itself.
        self.assertEqual(loop.nEvProcessed, self.nevents)

    def test_skip(self):
        first = 10 
        loop = Looper( self.outdir, config,
                       nEvents=None,
                       firstEvent=first,
                       nPrint=0,
                       timeReport=True)
        loop.loop()
        loop.write()
        # input file has 200 entries
        # we skip 10 entries, so we process 190.
        self.assertEqual(loop.nEvProcessed, self.nevents-first)

    def test_process_event(self):
        '''Test that indeed, calling loop.process(iev) raises
        TypeError if the events backend does not support indexing. 
        '''
        loop = Looper( self.outdir, config,
                       nEvents=None,
                       nPrint=0,
                       timeReport=True)
        self.assertRaises(TypeError, loop.process, 10)
        
       
if __name__ == '__main__':

    unittest.main()
