import unittest
import math
from ROOT import TLorentzVector
import numpy as np

from PhysicsTools.HeppyCore.particles.tlv.particle import Particle
from PhysicsTools.HeppyCore.configuration import Collider

from deltar import *

class TestDeltaR(unittest.TestCase):
    
    #----------------------------------------------------------------------
    def setUp(self):
        """maps the space with particles"""
        self.ptcs = {}
        for eta in range(-30, 30, 2):
            eta /= 10.
            for phi in range(-30, 30, 2):
                phi /= 10.
                tlv = TLorentzVector()
                tlv.SetPtEtaPhiM(10, eta, phi, 0)
                self.ptcs[(eta, phi)] = Particle(1, 0, tlv)
    
    #----------------------------------------------------------------------
    def test_deltaPhi(self):
        """Test that the deltaPhi function works properly around pi"""
        dphi = deltaPhi(math.pi-0.1, -math.pi+0.1)
        self.assertAlmostEqual(dphi, -0.2)
    
    #----------------------------------------------------------------------
    def test_deltaR2(self):
        """Test that the deltaR2 method properly uses either eta or
        theta depending on the collider configuration
        """
        Collider.BEAMS = 'pp'
        tlv1 = TLorentzVector()
        tlv1.SetPtEtaPhiM(10, 1.1, 0, 0)
        tlv2 = TLorentzVector()
        tlv2.SetPtEtaPhiM(10, 1.2, 0, 0)
        ptc1 = Particle(1, 1, tlv1)        
        ptc2 = Particle(1, 1, tlv2)
        dR = math.sqrt( deltaR2(ptc1, ptc2))
        self.assertAlmostEqual(dR, 0.1)
        
        Collider.BEAMS = 'ee'
        tlv1 = TLorentzVector()
        tlv1.SetPtEtaPhiM(10, 1.1, 0, 0)
        tlv1.SetTheta(1.1)
        tlv2 = TLorentzVector()
        tlv2.SetPtEtaPhiM(10, 1.2, 0, 0)
        tlv2.SetTheta(1.2)
        ptc1 = Particle(1, 1, tlv1)        
        ptc2 = Particle(1, 1, tlv2)
        dR = math.sqrt( deltaR2(ptc1, ptc2))
        self.assertAlmostEqual(dR, 0.1)
        
    #----------------------------------------------------------------------
    def test_inConeCollection(self):
        ptc0 = self.ptcs[(0, 0)]
        # very small cone. check that the pivot is
        # not included
        in_cone = inConeCollection(ptc0, self.ptcs.values(), 0.01)
        self.assertEqual(len(in_cone), 0)
        # four neighbours
        in_cone = inConeCollection(ptc0, self.ptcs.values(), 0.201)
        self.assertEqual(len(in_cone), 4)
        
    def test_cleanObjectCollection(self):
        # masking only one particle
        clean, dirty = cleanObjectCollection(self.ptcs.values(),
                                             [ self.ptcs[(0, 0)] ],
                                             0.01)
        self.assertEqual(dirty, [self.ptcs[0, 0]])
        self.assertEqual(len(clean), len(self.ptcs) - 1 )
        
        
if __name__ == '__main__':
    
    unittest.main()
