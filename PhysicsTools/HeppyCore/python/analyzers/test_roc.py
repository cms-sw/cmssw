import unittest
import numpy as np

from roc import cms_roc

class TestROC(unittest.TestCase):
    
    def test_all_wps(self):
        for b_eff, fake_eff in cms_roc.sig_bgd_points:
            cms_roc.set_working_point(b_eff)
            found = []
            for i in range(50000):
                found.append(cms_roc.is_b_tagged(True))
            self.assertAlmostEqual(np.average(found), b_eff, 2)
            fake = []
            for i in range(50000):
                fake.append(cms_roc.is_b_tagged(False))
            self.assertAlmostEqual(np.average(fake), fake_eff, 2)
            
                
if __name__ == '__main__':
    unittest.main()
