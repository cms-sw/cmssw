import os
import PhysicsTools.HeppyCore.framework.config as cfg


# input component 
# several input components can be declared,
# and added to the list of selected components
inputSample = cfg.MCComponent(
    'test_component',
    files = ['tt.root'],
    # tree_name = 'test_tree
    triggers = [], 
    xSection = 259, 
    nGenEvents = 1, 
    effCorrFactor = 1
    )

selectedComponents  = [inputSample]

from PhysicsTools.Heppy.analyzers.JSONAnalyzer import JSONAnalyzer
json = cfg.Analyzer(
    JSONAnalyzer
    )

from PhysicsTools.Heppy.analyzers.JetAnalyzer import JetAnalyzer
jets = cfg.Analyzer(
    JetAnalyzer,
    jetCol = 'slimmedJets',
    # cmg jet input collection
    # pt threshold
    jetPt = 30,
    # eta range definition
    jetEta = 5.0,
    # seed for the btag scale factor
    # btagSFseed = 123456,
    # if True, the PF and PU jet ID are not applied, and the jets get flagged
    relaxJetId = False,
    btagSFseed = 0xdeadbeef
    )

# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    json,
    jets
    ] )

# finalization of the configuration object. 
from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
config = cfg.Config( components = selectedComponents,
                     sequence = sequence, 
                     events_class = Events)

print config 
