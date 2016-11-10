from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.particles.tlv.particle import Particle as Recoil

from ROOT import TLorentzVector

mass = {23: 91, 25: 125}

class RecoilBuilder(Analyzer):
    '''Computes the 4 momentum recoiling agains a selection of particles.
    
    Example: 
    from PhysicsTools.HeppyCore.analyzers.RecoilBuilder import RecoilBuilder
    recoil = cfg.Analyzer(
      RecoilBuilder,
      output = 'recoil',
      sqrts = 240.,
      to_remove = 'zeds_legs'
    ) 

    * output : the recoil "particle" is stored in this collection. 
    
    * sqrts : energy in the center of mass system.

    * to_remove : collection of particles to be subtracted to the initial p4.
    if to_remove is set to the whole collection of reconstructed particles
    in the event, the missing p4 is computed.

    '''
    
    def process(self, event):
        sqrts = self.cfg_ana.sqrts
        to_remove = getattr(event, self.cfg_ana.to_remove) 
        recoil_p4 = TLorentzVector(0, 0, 0, sqrts)
        for ptc in to_remove:
            recoil_p4 -= ptc.p4()
        recoil = Recoil(0, 0, recoil_p4, 1) 
        setattr(event, self.cfg_ana.output, recoil)
                
