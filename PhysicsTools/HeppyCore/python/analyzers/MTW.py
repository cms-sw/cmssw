from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
import math

class MTW(Analyzer):
    
    def process(self, event):
        ele = getattr(event, self.cfg_ana.electron)
        mu = getattr(event, self.cfg_ana.muon)

        lepton = ele[0] if len(ele)==1 else mu[0]

        met = getattr(event, self.cfg_ana.met)
        mtw = math.sqrt(2.*lepton.pt()*met.pt()*(1-math.cos(lepton.phi() - met.phi() )))
        
        setattr(event, self.instance_label, mtw)

