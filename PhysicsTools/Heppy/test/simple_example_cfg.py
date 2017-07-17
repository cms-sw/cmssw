import os
import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.Heppy.utils.miniAodFiles import miniAodFiles

# set to True if you want several parallel processes
multi_thread = False

# input component 
# several input components can be declared,
# and added to the list of selected components
inputSample = cfg.MCComponent(
    'test_component',
    files = miniAodFiles(),  # returns a list of file for this release
    # a list of local or xrootd files can be specified by hand.
    )

if 'RelValZMM' not in inputSample.files[0]:
    print '''WARNING: this tutorial is supposed to run on Z->mumu events.
Do not expect meaningful results for this sample:'''
    print inputSample

if multi_thread: 
    inputSample.splitFactor = len(inputSample.files)

selectedComponents  = [inputSample]

# a very simple muon analyzer
# read miniaod muons and wrap them in python muons
from PhysicsTools.Heppy.analyzers.examples.SimpleMuonAnalyzer import SimpleMuonAnalyzer
muons = cfg.Analyzer(
    SimpleMuonAnalyzer,
    'muons',
    )

from PhysicsTools.Heppy.analyzers.examples.ResonanceBuilder import ResonanceBuilder
dimuons = cfg.Analyzer(
    ResonanceBuilder,
    'dimuons',
    leg_collection = 'muons',
    filter_func = lambda x : True, 
    pdgid = 23
    )


# a very simple jet analyzer
# read miniaod jets and wrap them in python jets
from PhysicsTools.Heppy.analyzers.examples.SimpleJetAnalyzer import SimpleJetAnalyzer
all_jets = cfg.Analyzer(
    SimpleJetAnalyzer,
    'all_jets',
    njets = 4, 
    filter_func = lambda x : True
    )

# filtering could be done in the SimpleJetAnalyzer above. 
# here, we illustrate the use of the generic Filter module
from PhysicsTools.HeppyCore.analyzers.Filter import Filter
sel_jets = cfg.Analyzer(
    Filter,
    'jets',
    input_objects = 'all_jets',
    filter_func = lambda x : x.pt()>30 
    )


# a simple tree with a Z candidate and the two leading jets (if any)
from PhysicsTools.Heppy.analyzers.examples.ZJetsTreeAnalyzer import ZJetsTreeAnalyzer
tree = cfg.Analyzer(
    ZJetsTreeAnalyzer
    )


# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [ 
        muons,
        dimuons,
        all_jets,
        sel_jets,
        tree
        ] )

# finalization of the configuration object. 
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
config = cfg.Config( components = selectedComponents,
                     sequence = sequence, 
                     services = [],
                     events_class = Events)

print config

if __name__ == '__main__':
    # can either run this configuration through heppy, 
    # or directly in python or ipython for easier development. 
    # try: 
    # 
    #   ipython -i simple_example_cfg.py
    # 
    from PhysicsTools.Heppy.physicsutils.LorentzVectors import LorentzVector

    from PhysicsTools.HeppyCore.framework.looper import Looper 
    looper = Looper( 'Loop', config, nPrint = 5, nEvents=100) 
    looper.loop()
    looper.write()

    # and now, let's play with the contents of the event
    print looper.event
    pz = LorentzVector()
    for imu, mu in enumerate(looper.event.muons): 
        print 'muon1', mu, 'abs iso=', mu.relIso()*mu.pt()
        pz += mu.p4()
    print 'z candidate mass = ', pz.M()

    # you can stay in ipython on a given event 
    # and paste more and more code as you need it until 
    # your code is correct. 
    # then put your code in an analyzer, and loop again. 

    def next():
        looper.process(looper.event.iEv+1)
