from heppy.framework.analyzer import Analyzer
from heppy.utils.deltar import matchObjectCollection, deltaR

import collections

class Matcher(Analyzer):
    '''Particle matcher. 

    Works with any kind of object with a p4 function. 

    Simple example configuration: 
    
    from heppy_fcc.analyzers.Matcher import Matcher
    papas_jet_match = cfg.Analyzer(
      Matcher,
      instance_label = 'papas', 
      match_particles = 'gen_jets',
      particles = 'papas_jets'
    )

    particles: Name of the collection containing the particles to be matched. 
    match_particles: Name of the collection containing the particles where a match 
               is to be found. 

    In this particular case, each jet in "papas_jets" will end up with a new 
    attribute called "match". This attribute can be either the closest gen jet in the 
    "gen_jets" collection in case a gen_jet is found within delta R = 0.3, 
    or None in case a match cannot be found in this cone.

    More complex example configuration: 

    papas_particle_match_g2r = cfg.Analyzer(
      Matcher,
      instance_label = 'papas_g2r', 
      particles = 'gen_particles_stable',
      match_particles = [
        ('papas_rec_particles', None),
        ('papas_rec_particles', 211),
        ('papas_rec_particles', 130),
        ('papas_rec_particles', 22)
      ] 
      )

    In this case, each gen particle in gen_particles_stable will end up with the following 
    new attributes: 
      - "match"    : closest reconstructed particle in "papas_rec_particles", if any. 
      - "match_211": closest reconstructed particle of pdgId 211 in "papas_rec_particles", 
                     if any. 
      - etc. 

    '''
    
    
    def beginLoop(self, setup):
        super(Matcher, self).beginLoop(setup)
        self.match_collections = []
        if isinstance( self.cfg_ana.match_particles, basestring):
            self.match_collections.append( (self.cfg_ana.match_particles, None) )
        else:
            self.match_collections = self.cfg_ana.match_particles
        
    def process(self, event):
        particles = getattr(event, self.cfg_ana.particles)
        # match_particles = getattr(event, self.cfg_ana.match_particles)
        for collname, pdgid in self.match_collections:
            match_ptcs = getattr(event, collname)
            match_ptcs_filtered = match_ptcs
            if pdgid is not None:
                match_ptcs_filtered = [ptc for ptc in match_ptcs
                                       if ptc.pdgid()==pdgid]
            pairs = matchObjectCollection(particles, match_ptcs_filtered,
                                          0.3**2)
            for ptc in particles:
                matchname = 'match'
                if pdgid: 
                    matchname = 'match_{pdgid}'.format(pdgid=pdgid)
                match = pairs[ptc]
                setattr(ptc, matchname, match)
                if match:
                    drname = 'dr'
                    if pdgid:
                        drname = 'dr_{pdgid}'.format(pdgid=pdgid)
                    dr = deltaR(ptc.theta(), ptc.phi(),
                                match.theta(), match.phi())
                    setattr(ptc, drname, dr)
                    # print dr, ptc, match
