import unittest
import tempfile
import copy
import os
import shutil

import PhysicsTools.HeppyCore.framework.context as context

if context.name == 'fcc':

    from analysis_ee_Z_cfg import config
    from PhysicsTools.HeppyCore.test.plot_ee_Z import plot
    from PhysicsTools.HeppyCore.framework.looper import Looper
    from ROOT import TFile

    import logging
    logging.getLogger().setLevel(logging.ERROR)

    import PhysicsTools.HeppyCore.statistics.rrandom as random

    class TestAnalysis_ee_Z(unittest.TestCase):

        def setUp(self):
            random.seed(0xdeadbeef)
            self.outdir = tempfile.mkdtemp()
            fname = '/'.join([os.environ['HEPPY'],
                              'test/data/ee_Z_ddbar.root'])
            config.components[0].files = [fname]
            self.looper = Looper( self.outdir, config,
                                  nEvents=100,
                                  nPrint=0,
                                  timeReport=True)
            import logging
            logging.disable(logging.CRITICAL)

        def tearDown(self):
            shutil.rmtree(self.outdir)
            logging.disable(logging.NOTSET)

        def test_analysis(self):
            '''Check for an almost perfect match with reference.
            Will fail if physics algorithms are modified,
            so should probably be removed from test suite,
            or better: be made optional. 
            '''
            self.looper.loop()
            self.looper.write()
            rootfile = '/'.join([self.outdir,
                                'PhysicsTools.HeppyCore.analyzers.GlobalEventTreeProducer.GlobalEventTreeProducer_1/tree.root'])
            mean, sigma = plot(rootfile)
            self.assertAlmostEqual(mean, 93.6, 1)
            self.assertAlmostEqual(sigma, 11.2, 1)



if __name__ == '__main__':

    unittest.main()
