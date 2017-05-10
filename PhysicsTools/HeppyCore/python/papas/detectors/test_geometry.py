import unittest
from geometry import *
from CMS import CMS

class TestCylinder(unittest.TestCase):
    def test_cylinders(self):
        cyl1 = SurfaceCylinder('cyl1', 1, 2)
        cyl2 = SurfaceCylinder('cyl2', 0.7, 1.5)
        subcyl = VolumeCylinder( 'subcyl', 1 ,2, 0.7, 1.5 ) 
        self.assertEqual(subcyl.inner.rad, 0.7)
        self.assertEqual(subcyl.outer.rad, 1.)
        self.assertEqual(subcyl.inner.z, 1.5)
        self.assertEqual(subcyl.outer.z, 2.)
        # inner cylinder larger than the outer one 
        self.assertRaises(ValueError,
                          VolumeCylinder, 'test', 1, 2, 1.1, 1.5 )
        # signature does not exist anymore
        self.assertRaises(TypeError,
                          VolumeCylinder, cyl2, cyl1 )
        # forgot name 
        self.assertRaises(ValueError,
                          VolumeCylinder, 1, 2, 0.9, 1.9)

    def test_print(self):
        cyl2 = SurfaceCylinder('cyl2', 0.7, 1.5)
        self.assertEqual(str(cyl2), 'SurfaceCylinder : cyl2, R= 0.70, z= 1.50')

class TestCMS(unittest.TestCase):
    def test_surfcyl_sorted(self):
        '''Make sure 
        - the surfaces are sorted by increasing radius
        - the z are increasing as well'''
        cms = CMS()
        radii = [cyl.rad for cyl in cms.cylinders()]
        self.assertEqual( radii, sorted(radii))
        zs = [cyl.z for cyl in cms.cylinders()]
        self.assertEqual( zs, sorted(zs))
        
if __name__ == '__main__':
    unittest.main()
