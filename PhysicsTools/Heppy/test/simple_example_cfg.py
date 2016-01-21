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
    inputSample.splitFactor = len(inputSamples.files)

selectedComponents  = [inputSample]

# a very simple muon analyzer
# read miniaod muons and wrap them in python muons
from PhysicsTools.Heppy.analyzers.examples.SimpleMuonAnalyzer import SimpleMuonAnalyzer
muons = cfg.Analyzer(
    SimpleMuonAnalyzer,
    'muons',
    )

# a very simple jet analyzer
# read miniaod jets and wrap them in python jets
from PhysicsTools.Heppy.analyzers.examples.SimpleJetAnalyzer import SimpleJetAnalyzer
jets = cfg.Analyzer(
    SimpleJetAnalyzer,
    'jets',
    ptmin = 30. # minimum pt cut for considering the jet
    )

# a simple tree with a Z candidate and the two leading jets (if any)
from PhysicsTools.Heppy.analyzers.examples.SimpleTreeAnalyzer import SimpleTreeAnalyzer
tree = cfg.Analyzer(
    SimpleTreeAnalyzer
    )


# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [ 
        muons,
        jets,
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
    from PhysicsTools.HeppyCore.framework.looper import Looper 
    looper = Looper( 'Loop', config, nPrint = 5, nEvents=100) 
    looper.loop()
    looper.write()

    def next():
        looper.process(looper.event.iEv+1)
