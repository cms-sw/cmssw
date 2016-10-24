from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.particles.fcc.particle import Particle 

import math
from PhysicsTools.HeppyCore.papas.simulator import Simulator
from PhysicsTools.HeppyCore.papas.vectors import Point
from PhysicsTools.HeppyCore.papas.pfobjects import Particle as PFSimParticle
from PhysicsTools.HeppyCore.papas.toyevents import particles
from PhysicsTools.HeppyCore.display.core import Display
from PhysicsTools.HeppyCore.display.geometry import GDetector
from PhysicsTools.HeppyCore.display.pfobjects import GTrajectories

from ROOT import TLorentzVector, TVector3

        
class PapasPF(Analyzer):
    '''Runs PAPAS, the PArametrized Particle Simulation.

    Example configuration: 

    from PhysicsTools.HeppyCore.analyzers.PapasPF import PapasPF
    from PhysicsTools.HeppyCore.papas.detectors.CMS import CMS
    papas = cfg.Analyzer(
        PapasPF,
        instance_label = 'papas',              
        detector = CMS(),
        gen_particles = 'gen_particles_stable',
        sim_particles = 'sim_particles',
        rec_particles = 'rec_particles',
        display = False,                   
        verbose = False
    )

    detector:      Detector model to be used. 
    gen_particles: Name of the input gen particle collection
    sim_particles: Name extension for the output sim particle collection. 
                   Note that the instance label is prepended to this name. 
                   Therefore, in this particular case, the name of the output 
                   sim particle collection is "papas_sim_particles".
    rec_particles: Name extension for the output reconstructed particle collection.
                   Same comments as for the sim_particles parameter above. 
    display      : Enable the event display
    verbose      : Enable the detailed printout.
    '''

    def __init__(self, *args, **kwargs):
        super(PapasPF, self).__init__(*args, **kwargs)
                
    def process(self, event):
        ecal = event.ECALclusters
        hcal = event.HCALclusters
        tracks = event.tracks
