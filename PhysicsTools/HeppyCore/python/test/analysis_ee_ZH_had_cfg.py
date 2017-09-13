'''Example configuration file for an ee->ZH analysis in the 4 jet channel,
with the FCC-ee

While studying this file, open it in ipython as well as in your editor to 
get more information: 

ipython
from analysis_ee_ZH_had_cfg import * 

'''

import os
import copy
import PhysicsTools.HeppyCore.framework.config as cfg

from PhysicsTools.HeppyCore.framework.event import Event
Event.print_patterns=['*jet*', 'bquarks', '*higgs*',
                      '*zed*', '*lep*']

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
Collider.SQRTS = 240.

# input definition
comp = cfg.Component(
    'ee_ZH_Z_Hbb',
    files = [
        'ee_ZH_Z_Hbb.root'
    ]
)
selectedComponents = [comp]

# read FCC EDM events from the input root file(s)
# do help(Reader) for more information
from PhysicsTools.HeppyCore.analyzers.fcc.Reader import Reader
source = cfg.Analyzer(
    Reader,
    gen_particles = 'GenParticle',
    gen_vertices = 'GenVertex'
)

# the papas simulation and reconstruction sequence
from PhysicsTools.HeppyCore.test.papas_cfg import papas_sequence, detector, papas

# Use a Filter to select leptons from the output of papas simulation.
# Currently, we're treating electrons and muons transparently.
# we could use two different instances for the Filter module
# to get separate collections of electrons and muons
# help(Filter) for more information
from PhysicsTools.HeppyCore.analyzers.Filter import Filter
def is_lepton(ptc):
    return ptc.e()> 5. and abs(ptc.pdgid()) in [11, 13]

leptons = cfg.Analyzer(
    Filter,
    'sel_leptons',
    output = 'leptons',
    input_objects = 'rec_particles',
    filter_func = is_lepton 
)

# Compute lepton isolation w/r other particles in the event.
# help(IsolationAnalyzer) 
# help(isolation) 
# for more information
from PhysicsTools.HeppyCore.analyzers.IsolationAnalyzer import IsolationAnalyzer
from PhysicsTools.HeppyCore.particles.isolation import EtaPhiCircle
iso_leptons = cfg.Analyzer(
    IsolationAnalyzer,
    leptons = 'leptons',
    particles = 'rec_particles',
    iso_area = EtaPhiCircle(0.4)
)

# Select isolated leptons with a Filter
def is_isolated(lep):
    '''returns true if the particles around the lepton
    in the EtaPhiCircle defined above carry less than 30%
    of the lepton energy.'''
    return lep.iso.sume/lep.e()<0.3  # fairly loose

sel_iso_leptons = cfg.Analyzer(
    Filter,
    'sel_iso_leptons',
    output = 'sel_iso_leptons',
    input_objects = 'leptons',
    filter_func = is_isolated
)


##Rejecting events that contain a loosely isolated lepton
##
##Instead of using an event filter at this stage, we store in the tree
##the lepton with lowest energy (with the name lepton1)
##
##from PhysicsTools.HeppyCore.analyzers.EventFilter import EventFilter
##lepton_veto = cfg.Analyzer(
##    EventFilter,
##    'lepton_veto',
##    input_objects='sel_iso_leptons',
##    min_number=1,
##    veto=True
##)

# compute the missing 4-momentum
from PhysicsTools.HeppyCore.analyzers.RecoilBuilder import RecoilBuilder
missing_energy = cfg.Analyzer(
    RecoilBuilder,
    instance_label = 'missing_energy',
    output = 'missing_energy',
    sqrts = Collider.SQRTS,
    to_remove = 'rec_particles'
) 


# make 4 exclusive jets 
from PhysicsTools.HeppyCore.analyzers.fcc.JetClusterizer import JetClusterizer
jets = cfg.Analyzer(
    JetClusterizer,
    output = 'jets',
    particles = 'rec_particles',
    fastjet_args = dict( njets = 4)  
)

# make 4 gen jets with stable gen particles
genjets = cfg.Analyzer(
    JetClusterizer,
    output = 'genjets',
    particles = 'gen_particles_stable',
    fastjet_args = dict( njets = 4)  
)

# select b quarks for jet to parton matching
def is_bquark(ptc):
    '''returns True if the particle is an outgoing b quark,
    see
    http://home.thep.lu.se/~torbjorn/pythia81html/ParticleProperties.html
    '''
    return abs(ptc.pdgid()) == 5 and ptc.status() == 23
    
bquarks = cfg.Analyzer(
    Filter,
    'bquarks',
    output = 'bquarks',
    input_objects = 'gen_particles',
    filter_func =is_bquark
)

# match genjets to b quarks 
from PhysicsTools.HeppyCore.analyzers.Matcher import Matcher
genjet_to_b_match = cfg.Analyzer(
    Matcher,
    match_particles = 'bquarks',
    particles = 'genjets',
    delta_r = 0.4
    )

# match jets to genjets (so jets are matched to b quarks through gen jets)
jet_to_genjet_match = cfg.Analyzer(
    Matcher,
    match_particles='genjets',
    particles='rescaled_jets',
    delta_r=0.5
)

# rescale the jet energy taking according to initial p4
from PhysicsTools.HeppyCore.analyzers.examples.zh_had.JetEnergyComputer import JetEnergyComputer
compute_jet_energy = cfg.Analyzer(
    JetEnergyComputer,
    output_jets='rescaled_jets',
    input_jets='jets',
    sqrts=Collider.SQRTS
    )

# parametrized b tagging with CMS performance.
# the performance of other detectors can be supplied
# in the roc module
# cms_roc is a numpy array, so one can easily scale
# the cms performance, help(numpy.array) for more info.
from PhysicsTools.HeppyCore.analyzers.ParametrizedBTagger import ParametrizedBTagger
from PhysicsTools.HeppyCore.analyzers.roc import cms_roc
cms_roc.set_working_point(0.7)
btag = cfg.Analyzer(
    ParametrizedBTagger,
    input_jets='rescaled_jets',
    roc=cms_roc
)

# reconstruction of the H and Z resonances.
# for now, use for the Higgs the two b jets with the mass closest to mH
# the other 2 jets are used for the Z.
# implement a chi2? 
from PhysicsTools.HeppyCore.analyzers.examples.zh_had.ZHReconstruction import ZHReconstruction
zhreco = cfg.Analyzer(
    ZHReconstruction,
    output_higgs='higgs',
    output_zed='zed', 
    input_jets='rescaled_jets'
)

# simple cut flow printout
from PhysicsTools.HeppyCore.analyzers.examples.zh_had.Selection import Selection
selection = cfg.Analyzer(
    Selection,
    input_jets='rescaled_jets', 
    log_level=logging.INFO
)

# Analysis-specific ntuple producer
# please have a look at the ZHTreeProducer class
from PhysicsTools.HeppyCore.analyzers.examples.zh_had.TreeProducer import TreeProducer
tree = cfg.Analyzer(
    TreeProducer,
    misenergy = 'missing_energy', 
    jets='rescaled_jets',
    higgs='higgs',
    zed='zed',
    leptons='sel_iso_leptons'
)

# definition of the sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence(
    source,
    papas_sequence, 
    leptons,
    iso_leptons,
    sel_iso_leptons,
#    lepton_veto, 
    jets,
    compute_jet_energy, 
    bquarks,
    genjets, 
    genjet_to_b_match,
    jet_to_genjet_match, 
    btag,
    missing_energy, 
    selection, 
    zhreco, 
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
    usage = '''usage: python analysis_ee_ZH_had_cfg.py [ievent]
    
    Provide ievent as an integer, or loop on the first events.
    You can also use this configuration file in this way: 
    
    PhysicsTools.HeppyCore.loop.py OutDir/ analysis_ee_ZH_had_cfg.py -f -N 100 
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
                   nPrint=10,
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
