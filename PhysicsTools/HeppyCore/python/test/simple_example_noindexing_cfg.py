import os
import copy 
import PhysicsTools.HeppyCore.framework.config as cfg

from PhysicsTools.HeppyCore.framework.chain_noindexing import ChainNoIndexing as Events

import logging
logging.basicConfig(level=logging.INFO)


# input component 
# several input components can be declared,
# and added to the list of selected components
inputSample = cfg.Component(
    'test_component',
    # create the test file by running
    # python create_tree.py
    files = [os.path.abspath('test_tree.root')],
    )

selectedComponents  = [inputSample]

from PhysicsTools.HeppyCore.analyzers.examples.simple.Printer import Printer
printer = cfg.Analyzer(
    Printer
    )

# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    printer,
] )


# finalization of the configuration object. 
config = cfg.Config( components = selectedComponents,
                     sequence = sequence,
                     services = [], 
                     events_class = Events )

# print config 
