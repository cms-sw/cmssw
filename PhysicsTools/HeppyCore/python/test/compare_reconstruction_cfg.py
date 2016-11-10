import os
import copy
import PhysicsTools.HeppyCore.framework.config as cfg

import logging

#import sys
# next 2 lines necessary to deal with reimports from ipython
logging.shutdown()
reload(logging)
logging.basicConfig(level=logging.WARNING)

comp = cfg.Component(
    'example',
    files = [
        'ee_ZH_Zmumu_Hbb.root'
    ]
)
selectedComponents = [comp]

from PhysicsTools.HeppyCore.analyzers.fcc.Reader import Reader
source = cfg.Analyzer(
    Reader,
    gen_particles = 'GenParticle',
)

# Use a Filter to select stable gen particles for simulation
# from the output of "source" 
# help(Filter) for more information
from PhysicsTools.HeppyCore.analyzers.Filter import Filter
gen_particles_stable = cfg.Analyzer(
    Filter,
    output = 'gen_particles_stable',
    # output = 'particles',
    input_objects = 'gen_particles',
    filter_func = lambda x : x.status()==1 and x.pdgid() not in [12,14,16] and x.pt()>0.1
)

from ROOT import gSystem
gSystem.Load("libdatamodelDict")
from EventStore import EventStore as Events
#from PhysicsTools.HeppyCore.framework.eventsgen import Events


#Run simulation (and include the original reconstruction of particles)
from PhysicsTools.HeppyCore.analyzers.PapasSim import PapasSim
from PhysicsTools.HeppyCore.papas.detectors.CMS import CMS
papas = cfg.Analyzer(
    PapasSim,
    instance_label = 'papas',
    detector = CMS(),
    gen_particles = 'gen_particles_stable',
    sim_particles = 'sim_particles',
    merged_ecals = 'ecal_clusters',
    merged_hcals = 'hcal_clusters',
    tracks = 'tracks',
    rec_particles = 'sim_rec_particles', # optional - will only do a simulation reconstruction if a anme is provided
    output_history = 'history_nodes',     
    display_filter_func = lambda ptc: ptc.e()>1.,
    display = False,
    verbose = True
)

#make connected blocks of tracks/clusters
from PhysicsTools.HeppyCore.analyzers.PapasPFBlockBuilder import PapasPFBlockBuilder
pfblocks = cfg.Analyzer(
    PapasPFBlockBuilder,
    tracks = 'tracks', 
    ecals = 'ecal_clusters', 
    hcals = 'hcal_clusters', 
    history = 'history_nodes', 
    output_blocks = 'reconstruction_blocks'    
)

#reconstruct particles
from PhysicsTools.HeppyCore.analyzers.PapasPFReconstructor import PapasPFReconstructor
pfreconstruct = cfg.Analyzer(
    PapasPFReconstructor,
    instance_label = 'papas_PFreconstruction', 
    detector = CMS(),
    input_blocks = 'reconstruction_blocks',
    history = 'history_nodes',     
    output_particles_dict = 'particles_dict', 
    output_particles_list = 'particles_list'    
)

#compare orignal and new reconstructions
from PhysicsTools.HeppyCore.analyzers.PapasParticlesComparer import PapasParticlesComparer 
particlescomparer = cfg.Analyzer(
    PapasParticlesComparer ,
    particlesA = 'papas_PFreconstruction_particles_list',
    particlesB = 'papas_sim_rec_particles'
)

# and then particle reconstruction from blocks 

# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    source,
    gen_particles_stable,
    papas,
    pfblocks,
    pfreconstruct,
    particlescomparer
    ] )
 
config = cfg.Config(
    components = selectedComponents,
    sequence = sequence,
    services = [],
    events_class = Events
)




if __name__ == '__main__':
    import sys
    from PhysicsTools.HeppyCore.framework.looper import Looper

    import PhysicsTools.HeppyCore.statistics.rrandom as random
    random.seed(0xdeadbeef)

    def process(iev=None):
        if iev is None:
            iev = loop.iEvent
        loop.process(iev)
        if display:
            display.draw()

    def next():
        loop.process(loop.iEvent+1)
        if display:
            display.draw()            

    iev = None
    if len(sys.argv)==2:
        papas.display = True
        iev = int(sys.argv[1])
       
    loop = Looper( 'looper', config,
                   nEvents=1000,
                   nPrint=0,
                   firstEvent=0,
                   timeReport=True)
    simulation = None
    for ana in loop.analyzers: 
        if hasattr(ana, 'display'):
            simulation = ana
    display = getattr(simulation, 'display', None)
    simulator = getattr(simulation, 'simulator', None)
    
    if simulator: 
        detector = simulator.detector
    if iev is not None:
        for j in range(10000) :
            process(iev)
            pass
    else:
        loop.loop()
        loop.write()
