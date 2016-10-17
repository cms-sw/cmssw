import unittest

from Gun import *

class GunTestCase(unittest.TestCase):
    
    def test_particle(self):
        for i in range(1000):
            ptc = particle(211, -0.5, 0.5, 10, 10, flat_pt=True)
            self.assertAlmostEqual(ptc.pt(), 10.)

    def test_e_pt_not_same(self):
        for i in range(1000):
            ptc = particle(211, -0.5, 0.5, 10, 10, flat_pt=True)
            self.assertNotEqual(ptc.pt(), ptc.e())
        
if __name__ == '__main__':
    unittest.main()
