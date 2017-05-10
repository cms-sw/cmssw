from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.papas.pdt import particle_data
from PhysicsTools.HeppyCore.particles.tlv.particle import Particle 

import math
import PhysicsTools.HeppyCore.statistics.rrandom as random

from ROOT import TLorentzVector

def particle(pdgid, thetamin, thetamax, ptmin, ptmax, flat_pt=False):
    mass, charge = particle_data[pdgid]
    theta = random.uniform(thetamin, thetamax)
    phi = random.uniform(-math.pi, math.pi)
    energy = random.uniform(ptmin, ptmax)
    costheta = math.cos(math.pi/2. - theta)
    sintheta = math.sin(math.pi/2. - theta)
    tantheta = sintheta / costheta
    cosphi = math.cos(phi)
    sinphi = math.sin(phi)        
    if flat_pt:
        pt = energy
        momentum = pt / sintheta
        energy = math.sqrt(momentum**2 + mass**2)
    else:
        energy = max([energy, mass])
        momentum = math.sqrt(energy**2 - mass**2)
    tlv = TLorentzVector(momentum*sintheta*cosphi,
                         momentum*sintheta*sinphi,
                         momentum*costheta,
                         energy)
    return Particle(pdgid, charge, tlv) 
    

class Gun(Analyzer):
    
    def process(self, event):
        event.gen_particles = [particle(self.cfg_ana.pdgid, 
                                        self.cfg_ana.thetamin, 
                                        self.cfg_ana.thetamax,
                                        self.cfg_ana.ptmin, 
                                        self.cfg_ana.ptmax,
                                        flat_pt=self.cfg_ana.flat_pt)]
        event.gen_particles_stable = event.gen_particles
