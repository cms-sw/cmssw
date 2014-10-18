import copy

from ROOT import RochCor, RochCor2012, TLorentzVector
from CMGTools.RootTools.utils.cmsswRelease import isNewerThan

is2012 = isNewerThan('CMSSW_5_2_X')

class RochesterCorrections(object):
    
    def __init__(self):
        self.cor = RochCor()
        self.cor2012 = RochCor2012()

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
        

# Below, Mike's C++ code from the HZZ analysis
##     unsigned int run = iEvent.id().run(); 

##     for (unsigned int i = 0; i < nsrc; ++i) {
##         T mu = (*src)[i];
## 	TLorentzVector p4(mu.px(),mu.py(),mu.pz(),mu.energy());


##         if (run <100 && !is55X) { //Monte Carlo 2011
## 	  corrector_.momcor_mc(p4, mu.charge(), 0.0, 0);
## 	}
## 	else if  (run <100 && is55X) {
## 	  corrector12_.momcor_mc(p4, mu.charge(), 0.0, 0);
## 	}
## 	else if (run>100&&run<=180252) { //2011 Data
##             corrector_.momcor_data(p4, mu.charge(), 0.0, run <= 173692 ? 0 : 1);
## 	}	 
## 	else  if (run>190000) { //2012 Data
##             corrector12_.momcor_data(p4, mu.charge(), 0.0, 0.0);
## 	}	 

## 	math::XYZTLorentzVector newP4(p4.Px(),p4.Py(),p4.Pz(),p4.Energy());
## 	mu.setP4(newP4);


##         out->push_back(mu);

