import os
import PhysicsTools.HeppyCore.framework.config as cfg

# input component 
# several input components can be declared,
# and added to the list of selected components
inputSample = cfg.Component(
    'test_component',
    files = ['test_tree.root'],
    # tree_name = 'test_tree'
    )

selectedComponents  = [inputSample]

printer = cfg.Analyzer(
    "Printer"
    )

tree = cfg.Analyzer(
    "SimpleTreeProducer",
    tree_name = 'tree',
    tree_title = 'A test tree'
    )

# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    printer,
    tree
    ] )

# finalization of the configuration object. 
config = cfg.Config( components = selectedComponents,
                     sequence = sequence )

print config 
