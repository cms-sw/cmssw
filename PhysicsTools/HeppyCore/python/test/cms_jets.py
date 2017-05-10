import os
import copy
import PhysicsTools.HeppyCore.framework.config as cfg

debug = False

if debug:
    print 'DEBUG MODE IS ON!'

comp = cfg.Component(
    'singlepi',
    files = ['/gridgroup/cms/cbernet/data/singlePi_50k.root']
)

selectedComponents = [comp]

from PhysicsTools.HeppyCore.analyzers.cms.JetReader import JetReader
source = cfg.Analyzer(
    JetReader,
    gen_jets = 'ak4GenJetsNoNu',
    gen_jet_pt = 20, 
    jets = 'ak4PFJets', 
    jet_pt = 20,
    nlead = 2 
)

from PhysicsTools.HeppyCore.analyzers.Matcher import Matcher
jet_match = cfg.Analyzer(
    Matcher,
    match_particles = 'cms_jets',
    particles = 'gen_jets',
    delta_r = 0.3
    )

from PhysicsTools.HeppyCore.analyzers.JetTreeProducer import JetTreeProducer
jet_tree = cfg.Analyzer(
    JetTreeProducer,
    tree_name = 'events',
    tree_title = 'jets',
    jets = 'gen_jets'
    )


from PhysicsTools.HeppyCore.framework.eventsfwlite import Events
  
if debug:
    comp = selectedComponents[0]
    comp.splitFactor =1 
    selectedComponents = [comp]



# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    source,
    jet_match,
    jet_tree
    ] )

    
config = cfg.Config(
    components = selectedComponents,
    sequence = sequence,
    services = [],
    events_class = Events
)

if __name__ == '__main__':
    print config
