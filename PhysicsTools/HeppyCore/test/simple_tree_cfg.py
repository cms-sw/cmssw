import os
import framework.config as cfg

inputSample = cfg.Component(
    'NameThatYouCanChoose',
    files = ['test_tree.root'],
    tree_name = 'test_tree'
    )

tree_producer = cfg.Analyzer(
    'SimpleTreeProducer',
    tree_name = 'simple_tree',
    tree_title = 'A simple tree'
    )

selectedComponents  = [inputSample]

sequence = cfg.Sequence( [
    tree_producer
    ] )

config = cfg.Config( components = selectedComponents,
                     sequence = sequence )
