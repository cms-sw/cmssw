
import os
import shutil

import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.HeppyCore.framework.chain import Chain as Events
import logging
logging.basicConfig(level=logging.INFO)

# input component 
# several input components can be declared,
# and added to the list of selected components

# os.system('python create_tree.py')
# shutil.copy('test_tree.root', 'test_tree_2.root')

inputSample = cfg.Component(
    'test_component',
    files = [os.path.abspath('test_tree.root'),
             os.path.abspath('test_tree_2.root')],
    splitFactor = 2
    )

selectedComponents  = [inputSample]


# add a random variable to the event 
from PhysicsTools.HeppyCore.analyzers.examples.simple.RandomAnalyzer import RandomAnalyzer
random = cfg.Analyzer(
    RandomAnalyzer
    )

# creating a simple output tree
from PhysicsTools.HeppyCore.analyzers.examples.simple.SimpleTreeProducer import SimpleTreeProducer
tree = cfg.Analyzer(
    SimpleTreeProducer,
    instance_label = 'tree',
    tree_name = 'tree',
    tree_title = 'A test tree'
    )


# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence([
        random,
        tree,
] )

from PhysicsTools.HeppyCore.framework.services.tfile import TFileService
output_rootfile = cfg.Service(
    TFileService,
    'myhists',
    fname='histograms.root',
    option='recreate'
)

services = [output_rootfile]

# finalization of the configuration object. 
config = cfg.Config( components = selectedComponents,
                     sequence = sequence,
                     services = services, 
                     events_class = Events )


