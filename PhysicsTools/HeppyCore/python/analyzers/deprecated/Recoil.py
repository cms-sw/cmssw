from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.particles.tlv.particle import Particle
from ROOT import TLorentzVector 


class Recoil(Analyzer):
    
    def process(self, event):
        initial = TLorentzVector()
        initial.SetXYZM(0,0,0,self.cfg_ana.sqrts)
        particles = getattr(event, self.cfg_ana.particles)
        visible_p4 = TLorentzVector()
        for ptc in particles: 
            if ptc.status()>1: #PF cand status=0 in CMS
                raise ValueError('are you sure? status='+str(ptc.status()) )
            visible_p4 += ptc.p4()
        recoil_p4 = initial - visible_p4
        recoil = Particle(0, 0, recoil_p4)
        visible = Particle(0, 0, visible_p4)
        setattr(event, '_'.join(['recoil', self.cfg_ana.instance_label]), recoil)
        setattr(event, '_'.join(['recoil_visible', self.cfg_ana.instance_label]), visible)
