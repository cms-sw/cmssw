import os
import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.Heppy.utils.miniAodFiles import miniAodFiles


# input component 
# several input components can be declared,
# and added to the list of selected components
inputSample = cfg.Component(
    'test_component',
    files = miniAodFiles(),
    )
inputSample.isMC = True

selectedComponents  = [inputSample]

from PhysicsTools.Heppy.analyzers.examples.SimpleJetAnalyzer import SimpleJetAnalyzer
jets = cfg.Analyzer(
    SimpleJetAnalyzer,
    ptmin = 50. 
    )

from PhysicsTools.Heppy.analyzers.examples.SimpleTreeAnalyzer import SimpleTreeAnalyzer
tree = cfg.Analyzer(
    SimpleTreeAnalyzer
    )


# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    jets,
    tree
    ] )

# finalization of the configuration object. 
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
config = cfg.Config( components = selectedComponents,
                     sequence = sequence, 
                     events_class = Events)

print config
import sys
print sys.modules.keys()
