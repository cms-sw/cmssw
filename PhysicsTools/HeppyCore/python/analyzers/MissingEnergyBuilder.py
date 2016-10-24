from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

from PhysicsTools.HeppyCore.particles.tlv.particle import Particle
from ROOT import TVector3, TLorentzVector

class MissingEnergyBuilder(Analyzer):
    
    def process(self, event):
        ptcs = getattr(event, self.cfg_ana.particles)
        sump3 = TVector3()
        charge = 0
        sume = 0 
        for ptc in ptcs: 
            sump3 += ptc.p3()
            charge += ptc.q()
        p4 = TLorentzVector()
        p4.SetVectM(-sump3, 0)
        missing = Particle(0, charge, p4)
        setattr(event, self.instance_label, missing)
