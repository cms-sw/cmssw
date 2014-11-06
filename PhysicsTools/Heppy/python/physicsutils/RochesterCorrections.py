import copy
import PhysicsTools.Heppy.loadlibs
from ROOT import heppy
from PhysicsTools.Heppy.utils.cmsswRelease import isNewerThan

is2012 = isNewerThan('CMSSW_5_2_0')

class RochesterCorrections(object):
    
    def __init__(self):
        self.cor = heppy.RochCor()
        self.cor2012 = heppy.RochCor2012()

    def corrected_p4( self, particle, run ):
        '''Returns the corrected p4 for a particle.

        The particle remains unchanged. 
        '''
        ptc = particle
        p4 = ptc.p4()
        tlp4 = TLorentzVector( p4.px(), p4.py(), p4.pz(), p4.energy() )
        cortlp4 = copy.copy(tlp4)
        if run<100:
            if is2012:
                self.cor2012.momcor_mc( cortlp4, ptc.charge(), 0.0, 0 )
            else:
                self.cor.momcor_mc( cortlp4, ptc.charge(), 0.0, 0 )
        else: # data
            if is2012:
                self.cor2012.momcor_data( cortlp4, ptc.charge(), 0.0, 0 )
            else:
                self.cor.momcor_data( cortlp4, ptc.charge(), 0.0, int(run>173692) )
        corp4 = p4.__class__( cortlp4.Px(), cortlp4.Py(), cortlp4.Pz(), cortlp4.Energy() )        
        return corp4

        
    def correct( self, particles, run ):
        '''Correct a list of particles.

        The p4 of each particle will change
        '''
        for ptc in particles: 
            corp4 = corrected_p4(ptc, run) 
            ptc.setP4( corp4 )


rochcor = RochesterCorrections() 
        

