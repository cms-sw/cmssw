import unittest
from PhysicsTools.HeppyCore.particles.tlv.resonance import Resonance2 as Resonance
from PhysicsTools.HeppyCore.particles.tlv.particle import Particle
from ROOT import TLorentzVector

class TestResonance(unittest.TestCase):

    def test_resonance(self):
        ptc1 = Particle(11, -1, TLorentzVector(1, 0, 0, 1))
        ptc2 = Particle(-11, 1, TLorentzVector(2, 0, 0, 2))
        reso = Resonance( ptc1, ptc2, 23 )
        self.assertEqual( reso._pid, 23 )
        self.assertEqual( reso.e(), 3 )
        self.assertEqual( reso.leg1(), ptc1 )
        self.assertEqual( reso.leg2(), ptc2 )
        self.assertEqual( reso.q(), 0 )
        self.assertEqual( reso.p4(), TLorentzVector(3,0,0,3) )

if __name__ == '__main__':
    unittest.main()
