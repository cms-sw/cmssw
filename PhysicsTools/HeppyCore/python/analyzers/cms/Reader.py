from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.particles.cms.particle import Particle 

import math

class CMSReader(Analyzer):
    
    def declareHandles(self):
        super(CMSReader, self).declareHandles()
        self.handles['gen_particles'] = AutoHandle(
            self.cfg_ana.gen_particles, 
            'std::vector<reco::GenParticle>'
            )
        self.read_pf = self.cfg_ana.pf_particles is not None
        if self.read_pf:
            self.handles['pf_particles'] = AutoHandle(
                self.cfg_ana.pf_particles, 
                'std::vector<reco::PFCandidate>'
                )

    def process(self, event):
        self.readCollections(event.input)
        store = event.input
        genp = self.handles['gen_particles'].product()
        gen_particles = map(Particle, genp)
        event.gen_particles = sorted( gen_particles,
                                      key = lambda ptc: ptc.e(), reverse=True )  
        event.gen_particles_stable = [ptc for ptc in event.gen_particles
                                      if ptc.status()==1 and 
                                      not math.isnan(ptc.e()) and
                                      ptc.e()>1e-5 and 
                                      ptc.pt()>1e-5 and
                                      not abs(ptc.pdgid()) in [12, 14, 16]]
        if self.read_pf:
            pfp = self.handles['pf_particles'].product()
            event.pf_particles = map(Particle, pfp)
        
