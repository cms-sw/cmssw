import unittest
import shutil
import tempfile
import os
import copy
from simple_example_cfg import config, stopper 
from PhysicsTools.HeppyCore.utils.testtree import create_tree, remove_tree
from PhysicsTools.HeppyCore.framework.heppy_loop import create_parser, main
from PhysicsTools.HeppyCore.framework.looper import Looper
from PhysicsTools.HeppyCore.framework.exceptions import UserStop
import PhysicsTools.HeppyCore.framework.context as context
from ROOT import TFile

import logging
logging.getLogger().setLevel(logging.ERROR)

class TestSimpleExample(unittest.TestCase):

    def setUp(self):
        self.fname = create_tree()
        rootfile = TFile(self.fname)
        self.nevents = rootfile.Get('test_tree').GetEntries()
        self.outdir = tempfile.mkdtemp()
        logging.disable(logging.CRITICAL)
        
    def tearDown(self):
        shutil.rmtree(self.outdir)
        logging.disable(logging.NOTSET)

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
        loop = Looper( self.outdir, config,
                       nEvents=None,
                       nPrint=0,
                       timeReport=True)
        loop.process(10)
        self.assertEqual(loop.event.input.var1, 10)
        loop.process(10)
        
    def test_userstop(self):
        config_with_stopper = copy.copy(config)
        config_with_stopper.sequence.insert(1, stopper)
        loop = Looper( self.outdir, config_with_stopper,
                       nEvents=None,
                       nPrint=0,
                       timeReport=True)
        self.assertRaises(UserStop, loop.process, 10)
  
    def test_rewrite(self):
        parser = create_parser()
        options, args = parser.parse_args()
        options.iEvent = None
        options.nprint = 0
        cfg = '/'.join( [ context.heppy_path, 
                          'test/simple_example_cfg.py' ] )
        main(options, [self.outdir, cfg], parser)
        options.force = True
        main(options, [self.outdir, cfg], parser)
        subdirs = os.listdir(self.outdir)
        self.assertEqual(len(subdirs), 2)

if __name__ == '__main__':

    unittest.main()
