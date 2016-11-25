import unittest
import tempfile
import copy
import os
import shutil

import PhysicsTools.HeppyCore.framework.context as context

if context.name == 'fcc':

    from analysis_ee_ZH_cfg import config
    from PhysicsTools.HeppyCore.test.plot_ee_ZH import plot
    from PhysicsTools.HeppyCore.framework.looper import Looper
    from ROOT import TFile

    import logging
    logging.getLogger().setLevel(logging.ERROR)

    import PhysicsTools.HeppyCore.statistics.rrandom as random

    def test_sorted(ptcs):
        from PhysicsTools.HeppyCore.configuration import Collider
        keyname = 'pt'
        if Collider.BEAMS == 'ee':
            keyname = 'e'
        pt_or_e = getattr(ptcs[0].__class__, keyname)
        values = [pt_or_e(ptc) for ptc in ptcs]
        return values == sorted(values, reverse=True)


    class TestAnalysis_ee_ZH(unittest.TestCase):

        def setUp(self):
            random.seed(0xdeadbeef)
            self.outdir = tempfile.mkdtemp()
            fname = '/'.join([os.environ['HEPPY'],
                              'test/data/ee_ZH_Zmumu_Hbb.root'])
            config.components[0].files = [fname]
            self.looper = Looper( self.outdir, config,
                                  nEvents=50,
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
                                'PhysicsTools.HeppyCore.analyzers.examples.zh.ZHTreeProducer.ZHTreeProducer_1/tree.root'])
            mean, sigma = plot(rootfile)
            self.assertAlmostEqual(mean, 120.7, 1)
            self.assertAlmostEqual(sigma, 20.3, 1)

        def test_analysis_sorting(self):
            self.looper.process(0)
            self.assertTrue(test_sorted(self.looper.event.rec_particles))

        


if __name__ == '__main__':

    unittest.main()
