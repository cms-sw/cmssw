import unittest
import os
import shutil

from PhysicsTools.HeppyCore.framework.config import *

class ConfigTestCase(unittest.TestCase):

    def test_analyzer(self):
        class Ana1(object):
            pass
        ana1 = Analyzer(
            Ana1,
            toto = '1',
            tata = 'a'
            )
        # checking that the analyzer name does not contain a slash, 
        # to make sure the output directory name does not contain a subdirectory
        self.assertTrue( '/' not in ana1.name )

    def test_MCComponent(self):
        DYJets = MCComponent(
            name = 'DYJets',
            files ='blah_mc.root',
            xSection = 3048.,
            nGenEvents = 34915945,
            triggers = ['HLT_MC'],
            vertexWeight = 1.,
            effCorrFactor = 1 )
        self.assertTrue(True)

    def test_config(self):
        class Ana1(object):
            pass
        ana1 = Analyzer(
            Ana1,
            toto = '1',
            tata = 'a'
            )
        comp1 = Component( 
            'comp1',
            files='*.root',
            triggers='HLT_stuff'
            )
        from PhysicsTools.HeppyCore.framework.chain import Chain as Events
        config = Config( components = [comp1],
                         sequence = [ana1], 
                         services = [],
                         events_class = Events )




if __name__ == '__main__':
    unittest.main()
