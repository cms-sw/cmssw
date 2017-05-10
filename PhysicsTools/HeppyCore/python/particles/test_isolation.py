import unittest
import math
import copy
from PhysicsTools.HeppyCore.particles.isolation import *
from PhysicsTools.HeppyCore.particles.tlv.particle import Particle
from ROOT import TLorentzVector

class TestIsolation(unittest.TestCase):

    def test_circle(self):
        circle = EtaPhiCircle(2)
        self.assertTrue( circle.is_inside(0, 0, 1.9, 0) )
        self.assertTrue( circle.is_inside(0, 0, 0., 1.9) )
        self.assertFalse( circle.is_inside(0, 0, 2.1, 0) )
        xory = math.sqrt(1.9**2/2.)
        self.assertTrue( circle.is_inside(0, 0, xory, xory) )
        xory = math.sqrt(2.1**2/2.)
        self.assertFalse( circle.is_inside(0, 0, xory, xory) )

    def test_iso1(self):
        p4 = TLorentzVector()
        p4.SetPtEtaPhiM(10, 0, 0, 0.105)
        lepton = Particle(13, 1, p4)

        p4 = TLorentzVector()
        p4.SetPtEtaPhiM(1, 0, 0, 0.105)
        ptc = Particle(211, 1, p4)

        # test iso calc
        computer = IsolationComputer([EtaPhiCircle(0.4)])
        iso = computer.compute(lepton, [ptc,ptc])
        self.assertEqual(iso.sumpt, 2*ptc.pt())
        self.assertEqual(iso.sume, 2*ptc.e())
        self.assertEqual(iso.num, 2)

        # test IsolationInfo addition
        iso2 = copy.copy(iso)
        iso2 += iso
        self.assertEqual(iso2.sumpt, 4*ptc.pt())
        self.assertEqual(iso2.sume, 4*ptc.e())
        self.assertEqual(iso2.num, 4)
       
        # test veto
        computer = IsolationComputer([EtaPhiCircle(0.4)], [EtaPhiCircle(0.1)])
        iso = computer.compute(lepton, [ptc])
        self.assertEqual(iso.sumpt, 0.)

        
if __name__ == '__main__':
    unittest.main()
