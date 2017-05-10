from PhysicsTools.HeppyCore.framework.analyzer import Analyzer

from PhysicsTools.HeppyCore.particles.tlv.met import MET
from ROOT import TLorentzVector 

class METBuilder(Analyzer):
    
    def process(self, event):
        particles = getattr(event, self.cfg_ana.particles)
        missingp4 = TLorentzVector()
        sumpt = 0 
        for ptc in particles:
            missingp4 += ptc.p4()
            sumpt += ptc.pt()
        missingp4 *= -1
        met = MET(missingp4, sumpt)
        setattr(event, self.instance_label, met)
