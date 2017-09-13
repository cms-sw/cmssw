'''Example configuration file for an ee->ZH->mumubb analysis in heppy, with the FCC-ee

While studying this file, open it in ipython as well as in your editor to 
get more information: 

ipython
from analysis_ee_ZH_cfg import * 
'''

import os
import copy
import PhysicsTools.HeppyCore.framework.config as cfg

from PhysicsTools.HeppyCore.framework.event import Event
Event.print_patterns=['*jet*', 'sum*']

import logging
# next 2 lines necessary to deal with reimports from ipython
logging.shutdown()
reload(logging)
logging.basicConfig(level=logging.WARNING)

# setting the random seed for reproducible results
import PhysicsTools.HeppyCore.statistics.rrandom as random
random.seed(0xdeadbeef)

# definition of the collider
from PhysicsTools.HeppyCore.configuration import Collider
Collider.BEAMS = 'ee'
Collider.SQRTS = 91.

# input definition
import glob
files = glob.glob(os.environ['HEPPY']+'/test/ee_Z_ddbar_*.root')
ee_Z_ddbar = cfg.Component(
    'ee_Z_ddbar',
    files = files 
    )
ee_Z_ddbar.splitFactor = len(ee_Z_ddbar.files)

ee_Z_bbbar = cfg.Component(
    'ee_Z_bbbar',
    files = [
        'ee_Z_bbbar.root'
    ]
)


selectedComponents = [ee_Z_ddbar]

# read FCC EDM events from the input root file(s)
# do help(Reader) for more information
from PhysicsTools.HeppyCore.analyzers.fcc.Reader import Reader
source = cfg.Analyzer(
    Reader,
    gen_particles = 'GenParticle',
    gen_vertices = 'GenVertex'
)

from PhysicsTools.HeppyCore.analyzers.P4SumBuilder import P4SumBuilder
sum_particles = cfg.Analyzer(
    P4SumBuilder, 
    output='sum_all_ptcs',
    #    particles='gen_particles_stable'
    particles='rec_particles'
)

sum_gen = cfg.Analyzer(
    P4SumBuilder, 
    output='sum_all_gen',
    particles='gen_particles_stable'
)


from PhysicsTools.HeppyCore.analyzers.GlobalEventTreeProducer import GlobalEventTreeProducer
zed_tree = cfg.Analyzer(
    GlobalEventTreeProducer, 
    sum_all='sum_all_ptcs', 
    sum_all_gen='sum_all_gen'
)


from PhysicsTools.HeppyCore.test.papas_cfg import gen_particles_stable, papas_sequence, detector, papas
from PhysicsTools.HeppyCore.test.jet_tree_cff import jet_tree_sequence


# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence(
    source,
    # gen_particles_stable, 
    papas_sequence,
    jet_tree_sequence('gen_particles_stable',
                      'rec_particles',
                      2, None),
    sum_particles,
    sum_gen, 
    zed_tree
    )

# Specifics to read FCC events 
from ROOT import gSystem
gSystem.Load("libdatamodelDict")
from EventStore import EventStore as Events

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
    usage = '''usage: python analysis_ee_ZH_cfg.py [ievent]
    
    Provide ievent as an integer, or loop on the first events.
    You can also use this configuration file in this way: 
    
    heppy_loop.py OutDir/ analysis_ee_ZH_cfg.py -f -N 100 
    '''
    if len(sys.argv)==2:
        papas.display = True
        try:
            iev = int(sys.argv[1])
        except ValueError:
            print usage
            sys.exit(1)
    elif len(sys.argv)>2: 
        print usage
        sys.exit(1)
            
        
    loop = Looper( 'looper', config,
                   nEvents=10,
                   nPrint=5,
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
        process(iev)
    else:
        loop.loop()
        loop.write()
