import unittest
from detectors.geometry import SurfaceCylinder
from pfobjects import Particle
from propagator import straight_line, helix
from vectors import LorentzVector, Point

class TestPropagator(unittest.TestCase):
    
    def test_straightline(self):
        origin = Point(0,0,0)
        cyl1 = SurfaceCylinder('cyl1', 1, 2)
        cyl2 = SurfaceCylinder('cyl2', 2, 1)

        particle = Particle( LorentzVector(1, 0, 1, 2.), origin, 0)
        straight_line.propagate_one( particle, cyl1 )
        straight_line.propagate_one( particle, cyl2 )
        self.assertEqual( len(particle.points), 3)
        # test extrapolation to barrel
        self.assertAlmostEqual( particle.points['cyl1'].Perp(), 1. )
        self.assertAlmostEqual( particle.points['cyl1'].Z(), 1. )
        # test extrapolation to endcap
        self.assertAlmostEqual( particle.points['cyl2'].Z(), 1. )
        
        # testing extrapolation to -z 
        particle = Particle( LorentzVector(1, 0, -1, 2.), origin, 0)
        # import pdb; pdb.set_trace()
        straight_line.propagate_one( particle, cyl1 )
        straight_line.propagate_one( particle, cyl2 )
        self.assertEqual( len(particle.points), 3)
        self.assertAlmostEqual( particle.points['cyl1'].Perp(), 1. )
        # test extrapolation to endcap
        self.assertAlmostEqual( particle.points['cyl1'].Z(), -1. )
        self.assertAlmostEqual( particle.points['cyl2'].Z(), -1. )

        # extrapolating from a vertex close to +endcap
        particle = Particle( LorentzVector(1, 0, 1, 2.),
                             Point(0,0,1.5), 0)
        straight_line.propagate_one( particle, cyl1 )
        self.assertAlmostEqual( particle.points['cyl1'].Perp(), 0.5 )
        
        # extrapolating from a vertex close to -endcap
        particle = Particle( LorentzVector(1, 0, -1, 2.),
                             Point(0,0,-1.5), 0)
        straight_line.propagate_one( particle, cyl1 )
        self.assertAlmostEqual( particle.points['cyl1'].Perp(), 0.5 )
        
        # extrapolating from a non-zero radius
        particle = Particle( LorentzVector(0, 0.5, 1, 2.),
                             Point(0,0.5,0), 0)
        straight_line.propagate_one( particle, cyl1 )
        self.assertAlmostEqual( particle.points['cyl1'].Perp(), 1. )
        self.assertAlmostEqual( particle.points['cyl1'].Z(), 1. )

        # extrapolating from a z outside the cylinder
        particle = Particle( LorentzVector(0, 0, -1, 2.),
                             Point(0,0,2.5), 0)
        straight_line.propagate_one( particle, cyl1 )
        self.assertFalse( 'cyl1' in particle.points ) 
        
        # extrapolating from a z outside the cylinder, negative
        particle = Particle( LorentzVector(0, 0, -1, 2.),
                             Point(0,0,-2.5), 0)
        straight_line.propagate_one( particle, cyl1 )
        self.assertFalse( 'cyl1' in particle.points ) 

        # extrapolating from a rho outside the cylinder
        particle = Particle( LorentzVector(0, 0, -1, 2.),
                             Point(0,1.1,0), 0)
        straight_line.propagate_one( particle, cyl1 )
        self.assertFalse( 'cyl1' in particle.points ) 
                
    def test_helix(self):
        cyl1 = SurfaceCylinder('cyl1', 1., 2.)
        cyl2 = SurfaceCylinder('cyl2', 2., 1.)
        field = 3.8
        particle = Particle( LorentzVector(2., 0, 1, 5),
                             Point(0., 0., 0.), -1)        
        debug_info = helix.propagate_one(particle, cyl1, field)
        particle = Particle( LorentzVector(0., 2, 1, 5),
                             Point(0., 0., 0.), -1)        
        debug_info = helix.propagate_one(particle, cyl1, field)

        
if __name__ == '__main__':
    unittest.main()
