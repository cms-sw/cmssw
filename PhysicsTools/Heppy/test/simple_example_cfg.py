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

json = cfg.Analyzer(
    "JSONAnalyzer"
    )

# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    json
    ] )

# finalization of the configuration object. 
config = cfg.Config( components = selectedComponents,
                     sequence = sequence )

print config 
