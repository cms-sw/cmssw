import unittest
import os
import copy
from heppy.particles.tlv.particle import Particle as TlvParticle
from heppy.particles.fcc.particle import Particle as FccParticle
from heppy.configuration import Collider
from ROOT import TLorentzVector, gSystem


class TestParticle(unittest.TestCase):

    def tearDown(self):
        Collider.BEAMS = 'pp'
    
    def test_root_particle_copy(self):
        '''Test that root-based particles can be created, deepcopied,
        and compared.'''
        ptc = TlvParticle(1, 1, TLorentzVector())
        ptc2 = copy.deepcopy(ptc)
        self.assertEqual(ptc, ptc2)
        
    def test_printout(self):
        '''Test that the particle printout is adapted to the collider
        beams.'''
        ptc = TlvParticle(1, 1, TLorentzVector())        
        Collider.BEAMS = 'pp'
        self.assertIn('pt', ptc.__repr__())
        Collider.BEAMS = 'ee'
        self.assertIn('theta', ptc.__repr__())
        Collider.BEAMS = 'pp'
        
    #----------------------------------------------------------------------
    def test_sort(self):
        """Test that particles are sorted by energy or by pT depending
        on the collider beams"""
        ptcs = [TlvParticle(1, 1, TLorentzVector(10, 0, 0, 11)), 
                TlvParticle(1, 1, TLorentzVector(0, 0, 11, 12))]
        Collider.BEAMS = 'ee'
        self.assertEqual(sorted(ptcs, reverse=True),
                         [ptcs[1], ptcs[0]])
        Collider.BEAMS = 'pp'
        self.assertEqual(sorted(ptcs, reverse=True),
                         [ptcs[0], ptcs[1]])
        
        
    #----------------------------------------------------------------------
    def test_fcc_particle(self):
        """Test that FCC particles can be copied and compared"""
        if not 'FCCEDM' in os.environ: 
            return 
        retcode = gSystem.Load("libdatamodelDict")
        # testing only if the FCC EDM is available
        self.assertEqual(retcode, 0)
        try:
            from EventStore import EventStore as Events
        except ImportError:
            return
        test_fcc_file = '/'.join([os.environ['HEPPY'],
                                  'test/data/ee_ZH_Zmumu_Hbb.root'])
        events = Events([test_fcc_file])
        event = next(iter(events))
        fccptc = event.get('GenParticle')
        ptcs = map(FccParticle, fccptc)
        ptc0_2 = copy.deepcopy(ptcs[0])
        self.assertEqual(ptc0_2, ptcs[0])
        

if __name__ == '__main__':
    unittest.main()
