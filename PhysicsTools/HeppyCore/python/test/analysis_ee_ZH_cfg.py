'''Example configuration file for an ee->ZH->mumubb analysis in heppy, with the FCC-ee

While studying this file, open it in ipython as well as in your editor to 
get more information: 

ipython
from analysis_ee_ZH_cfg import * 

'''

import os
import copy
import PhysicsTools.HeppyCore.framework.config as cfg
import PhysicsTools.HeppyCore.utils.pdebug

import logging
# next 2 lines necessary to deal with reimports from ipython
logging.shutdown()
reload(logging)
logging.basicConfig(level=logging.WARNING)

# setting the random seed for reproducible results
import PhysicsTools.HeppyCore.statistics.rrandom as random
random.seed(0xdeadbeef)

from ROOT import gSystem
gSystem.Load("libdatamodelDict")
from EventStore import EventStore as Events
import PhysicsTools.HeppyCore.utils.pdebug

# definition of the collider
from PhysicsTools.HeppyCore.configuration import Collider
Collider.BEAMS = 'ee'
Collider.SQRTS = 240.

# input definition
comp = cfg.Component(
    'ee_ZH_Zmumu_Hbb',
    files = [
        os.path.abspath('ee_ZH_Zmumu_Hbb.root')
    ]
)
selectedComponents = [comp]

#  Pdebugger
from PhysicsTools.HeppyCore.analyzers.PDebugger import PDebugger
pdebug = cfg.Analyzer(
    PDebugger,
    output_to_stdout = False,
    debug_filename = os.getcwd()+'/python_physics_debug.log' #optional argument
)
# read FCC EDM events from the input root file(s)
# do help(Reader) for more information
from PhysicsTools.HeppyCore.analyzers.fcc.Reader import Reader
source = cfg.Analyzer(
    Reader,
    gen_particles = 'GenParticle',
    gen_vertices = 'GenVertex'
)

from PhysicsTools.HeppyCore.test.papas_cfg import papas_sequence, detector, papas

# Use a Filter to select leptons from the output of papas simulation.
# Currently, we're treating electrons and muons transparently.
# we could use two different instances for the Filter module
# to get separate collections of electrons and muons
# help(Filter) for more information
from PhysicsTools.HeppyCore.analyzers.Filter import Filter
leptons_true = cfg.Analyzer(
    Filter,
    'sel_leptons',
    output = 'leptons_true',
    input_objects = 'rec_particles',
    filter_func = lambda ptc: ptc.e()>10. and abs(ptc.pdgid()) in [11, 13]
)

# Compute lepton isolation w/r other particles in the event.
# help(IsolationAnalyzer) for more information
from PhysicsTools.HeppyCore.analyzers.IsolationAnalyzer import IsolationAnalyzer
from PhysicsTools.HeppyCore.particles.isolation import EtaPhiCircle
iso_leptons = cfg.Analyzer(
    IsolationAnalyzer,
    leptons = 'leptons_true',
    particles = 'rec_particles',
    iso_area = EtaPhiCircle(0.4)
)

# Select isolated leptons with a Filter
# one can pass a function like this one to the filter:
def relative_isolation(lepton):
    sumpt = lepton.iso_211.sumpt + lepton.iso_22.sumpt + lepton.iso_130.sumpt
    sumpt /= lepton.pt()
    return sumpt
# ... or use a lambda statement as done below. 
sel_iso_leptons = cfg.Analyzer(
    Filter,
    'sel_iso_leptons',
    output = 'sel_iso_leptons',
    input_objects = 'leptons_true',
    # filter_func = relative_isolation
    filter_func = lambda lep : lep.iso.sumpt/lep.pt()<0.3 # fairly loose
)

# Building Zeds
# help(ResonanceBuilder) for more information
from PhysicsTools.HeppyCore.analyzers.ResonanceBuilder import ResonanceBuilder
zeds = cfg.Analyzer(
    ResonanceBuilder,
    output = 'zeds',
    leg_collection = 'sel_iso_leptons',
    pdgid = 23
)

# Computing the recoil p4 (here, p_initial - p_zed)
# help(RecoilBuilder) for more information
sqrts = Collider.SQRTS 

from PhysicsTools.HeppyCore.analyzers.RecoilBuilder import RecoilBuilder
recoil = cfg.Analyzer(
    RecoilBuilder,
    instance_label = 'recoil',
    output = 'recoil',
    sqrts = sqrts,
    to_remove = 'zeds_legs'
) 

missing_energy = cfg.Analyzer(
    RecoilBuilder,
    instance_label = 'missing_energy',
    output = 'missing_energy',
    sqrts = sqrts,
    to_remove = 'rec_particles'
) 

# Creating a list of particles excluding the decay products of the best zed.
# help(Masker) for more information
from PhysicsTools.HeppyCore.analyzers.Masker import Masker
particles_not_zed = cfg.Analyzer(
    Masker,
    output = 'particles_not_zed',
    input = 'rec_particles',
    mask = 'zeds_legs',
)

# Make jets from the particles not used to build the best zed.
# Here the event is forced into 2 jets to target ZH, H->b bbar)
# help(JetClusterizer) for more information
from PhysicsTools.HeppyCore.analyzers.fcc.JetClusterizer import JetClusterizer
jets = cfg.Analyzer(
    JetClusterizer,
    output = 'jets',
    particles = 'particles_not_zed',
    fastjet_args = dict( njets = 2)  
)

from PhysicsTools.HeppyCore.analyzers.ImpactParameter import ImpactParameter
btag = cfg.Analyzer(
    ImpactParameter,
    jets = 'jets',
    # num_IP = ("histo_stat_IP_ratio_bems.root","h_b"),
    # denom_IP = ("histo_stat_IP_ratio_bems.root","h_u"),
    # num_IPs = ("histo_stat_IPs_ratio_bems.root","h_b"),
    # denom_IPs = ("histo_stat_IPs_ratio_bems.root","h_u"),
    pt_min = 1, # pt threshold for charged hadrons in b tagging 
    dxy_max = 2e-3, # 2mm
    dz_max = 17e-2, # 17cm
    detector = detector
    )

# Build Higgs candidates from pairs of jets.
higgses = cfg.Analyzer(
    ResonanceBuilder,
    output = 'higgses',
    leg_collection = 'jets',
    pdgid = 25
)


# Just a basic analysis-specific event Selection module.
# this module implements a cut-flow counter
# After running the example as
#    heppy_loop.py Trash/ analysis_ee_ZH_cfg.py -f -N 100 
# this counter can be found in:
#    Trash/example/PhysicsTools.HeppyCore.analyzers.examples.zh.selection.Selection_cuts/cut_flow.txt
# Counter cut_flow :
#         All events                                     100      1.00    1.0000
#         At least 2 leptons                              87      0.87    0.8700
#         Both leptons e>30                               79      0.91    0.7900
# For more information, check the code of the Selection class,
from PhysicsTools.HeppyCore.analyzers.examples.zh.selection import Selection
selection = cfg.Analyzer(
    Selection,
    instance_label='cuts'
)

# Analysis-specific ntuple producer
# please have a look at the ZHTreeProducer class
from PhysicsTools.HeppyCore.analyzers.examples.zh.ZHTreeProducer import ZHTreeProducer
tree = cfg.Analyzer(
    ZHTreeProducer,
    zeds = 'zeds',
    jets = 'jets',
    higgses = 'higgses',
    recoil  = 'recoil',
    misenergy = 'missing_energy'
)

# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence(
    pdebug,
    source,
    papas_sequence, 
    leptons_true,
    iso_leptons,
    sel_iso_leptons,
    zeds,
    recoil,
    missing_energy,
    particles_not_zed,
    jets,
    btag,
    higgses,
    selection, 
    tree
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
                   nPrint=1,
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
        pass
    else:
        loop.loop()
        loop.write()
