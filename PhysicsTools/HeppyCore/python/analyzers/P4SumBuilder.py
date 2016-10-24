from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.particles.tlv.particle import Particle
from PhysicsTools.HeppyCore.particles.tlv.jet import Jet
from PhysicsTools.HeppyCore.particles.jet import JetConstituents

from ROOT import TLorentzVector

mass = {23: 91, 25: 125}

class P4SumBuilder(Analyzer):
    '''Computes the 4 momentum recoiling agains a selection of particles.
    
    Example: 
    from PhysicsTools.HeppyCore.analyzers.P4SumBuilder import P4SumBuilder
    recoil = cfg.Analyzer(
      P4SumBuilder,
      output = 'sum_ptc',
      particles = 'rec_particles'
    ) 

    * output : contains a single particle with a p4 equal to the
               sum p4 of all input particles.
    
    * particles : collection of input particles.
    '''
    
    def process(self, event):
        p4 = TLorentzVector()
        charge = 0
        pdgid = 0
        ptcs = getattr(event, self.cfg_ana.particles)
        jet = Jet(p4)
        constituents = JetConstituents()
        for ptc in ptcs:
            p4 += ptc.p4()
            charge += ptc.q()
            constituents.append(ptc)
        sumptc = Particle(pdgid, charge, p4)
        jet = Jet(p4)
        jet.constituents = constituents
        jet.constituents.sort()
        setattr(event, self.cfg_ana.output, jet)
                
