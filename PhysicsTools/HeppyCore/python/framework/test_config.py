import unittest
import os
import shutil
import copy

import config as cfg
from analyzer import Analyzer 

class ConfigTestCase(unittest.TestCase):

    def test_analyzer(self):
        ana1 = cfg.Analyzer(
            Analyzer,
            toto = '1',
            tata = 'a'
            )
        # checking that the analyzer name does not contain a slash, 
        # to make sure the output directory name does not contain a subdirectory
        self.assertTrue( '/' not in ana1.name )

    def test_MCComponent(self):
        DYJets = cfg.MCComponent(
            name = 'DYJets',
            files ='blah_mc.root',
            xSection = 3048.,
            nGenEvents = 34915945,
            triggers = ['HLT_MC'],
            vertexWeight = 1.,
            effCorrFactor = 1 )
        self.assertTrue(True)

    def test_config(self):
        ana1 = cfg.Analyzer(
            Analyzer,
            toto = '1',
            tata = 'a'
            )
        comp1 = cfg.Component( 
            'comp1',
            files='*.root',
            triggers='HLT_stuff'
            )
        from PhysicsTools.HeppyCore.framework.chain import Chain as Events
        config = cfg.Config( components = [comp1],
                            sequence = [ana1], 
                            services = [],
                            events_class = Events )

    def test_copy(self):
        ana1 = cfg.Analyzer(
            Analyzer,
            instance_label = 'inst1',
            toto = '1',
            )        
        ana2 = copy.copy(ana1)
        ana2.instance_label = 'inst2'
        ana2.toto2 = '2'
        self.assertTrue(ana2.name.endswith('analyzer.Analyzer_inst2'))
        self.assertEqual(ana2.toto2, '2')

    def test_sequence(self):
        seq = cfg.Sequence( 0, 1, 2 )
        self.assertEqual(seq, range(3))
        seq = cfg.Sequence( range(3) )
        self.assertEqual(seq, range(3))
        seq = cfg.Sequence( range(3), 3)
        self.assertEqual(seq, range(4))
        seq = cfg.Sequence( 'blah' )
        self.assertEqual(seq, ['blah'])
        self.assertRaises(ValueError, cfg.Sequence, dict(a=1) )
        
        
if __name__ == '__main__':
    unittest.main()
