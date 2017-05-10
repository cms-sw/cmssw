import os
import copy
import PhysicsTools.HeppyCore.framework.config as cfg

import logging
# next 2 lines necessary to deal with reimports from ipython
logging.shutdown()
reload(logging)
logging.basicConfig(level=logging.WARNING)

comp = cfg.Component(
    'example',
    #files = ['example.root']
    files = ['root://eospublic.cern.ch//eos/fcc/users/h/helsens/DelphesOutputs/ttbar_13TeV/FCCDelphesOutput_ttbar13TeV_1.root',
             'root://eospublic.cern.ch//eos/fcc/users/h/helsens/DelphesOutputs/ttbar_13TeV/FCCDelphesOutput_ttbar13TeV_2.root',
             'root://eospublic.cern.ch//eos/fcc/users/h/helsens/DelphesOutputs/ttbar_13TeV/FCCDelphesOutput_ttbar13TeV_3.root',
             'root://eospublic.cern.ch//eos/fcc/users/h/helsens/DelphesOutputs/ttbar_13TeV/FCCDelphesOutput_ttbar13TeV_4.root',

        #'/afs/cern.ch/user/h/helsens/FCCsoft/FCCSOFT/FCC/FCCSW/FCCDelphesOutput.root'
             ]
    #files = ['FCCDelphes_ClementOutput1.root']
)
selectedComponents = [comp]

from PhysicsTools.HeppyCore.analyzers.fcc.Reader import Reader
source = cfg.Analyzer(
    Reader,
    #gen_particles = 'genParticles',
    gen_jets = 'genJets',

    jets = 'jets',
    bTags = 'bTags',
    jetsToBTags = 'jetsToBTags',

    electrons = 'electrons',
    electronITags = 'electronITags',
    electronsToITags = 'electronsToITags',

    muons = 'muons',
    muonITags = 'muonITags',
    muonsToITags = 'muonsToITags',

    photons = 'photons',
    met = 'met',
)  

from ROOT import gSystem
gSystem.Load("libdatamodelDict")
from EventStore import EventStore as Events



from PhysicsTools.HeppyCore.analyzers.Filter import Filter
muons = cfg.Analyzer(
    Filter,
    'sel_muons',
    output = 'muons',
    input_objects = 'muons',
    filter_func = lambda ptc: ptc.pt()>30
)

from PhysicsTools.HeppyCore.analyzers.Filter import Filter
iso_muons = cfg.Analyzer(
    Filter,
    'sel_iso_muons',
    output = 'sel_iso_muons',
    input_objects = 'muons',
    filter_func = lambda ptc: ptc.iso.sumpt/ptc.pt()<0.2
)

from PhysicsTools.HeppyCore.analyzers.Filter import Filter
electrons = cfg.Analyzer(
    Filter,
    'sel_electrons',
    output = 'electrons',
    input_objects = 'electrons',
    filter_func = lambda ptc: ptc.pt()>30
)

from PhysicsTools.HeppyCore.analyzers.Filter import Filter
iso_electrons = cfg.Analyzer(
    Filter,
    'sel_iso_electrons',
    output = 'sel_iso_electrons',
    input_objects = 'electrons',
    filter_func = lambda ptc: ptc.iso.sumpt/ptc.pt()<0.1
)


jets_30 = cfg.Analyzer(
    Filter,
    'jets_30',
    output = 'jets_30',
    input_objects = 'jets',
    filter_func = lambda jet: jet.pt()>30.
)

from PhysicsTools.HeppyCore.analyzers.Matcher import Matcher
match_jet_electrons = cfg.Analyzer(
    Matcher,
    'electron_jets',
    delta_r = 0.2,
    match_particles = 'sel_iso_electrons',
    particles = 'jets_30'
)

sel_jets_electron = cfg.Analyzer(
    Filter,
    'sel_jets_noelecetron_30',
    output = 'sel_jets_noelectron_30',
    input_objects = 'jets_30',
    filter_func = lambda jet: jet.match is None
)


from PhysicsTools.HeppyCore.analyzers.Matcher import Matcher
match_muon_jets = cfg.Analyzer(
    Matcher,
    'muon_jets',
    delta_r = 0.2,
    match_particles = 'sel_iso_muons',
    particles = 'sel_jets_noelectron_30'
)

sel_jets_muon = cfg.Analyzer(
    Filter,
    'sel_jets_nomuon_30',
    output = 'sel_jets_noelectronnomuon_30',
    input_objects = 'sel_jets_noelectron_30',
    filter_func = lambda jet: jet.match is None
)


from PhysicsTools.HeppyCore.analyzers.examples.ttbar.BTagging import BTagging
btagging = cfg.Analyzer(
    BTagging,
    'b_jets_30',
    output = 'b_jets_30',
    input_objects = 'sel_jets_noelectronnomuon_30',
    filter_func = lambda jet : jet.tags['bf']>0.
)


from PhysicsTools.HeppyCore.analyzers.M3Builder import M3Builder
m3 = cfg.Analyzer(
    M3Builder,
    instance_label = 'm3',
    jets = 'sel_jets_noelectronnomuon_30', 
    filter_func = lambda x : x.pt()>30.
)

from PhysicsTools.HeppyCore.analyzers.MTW import MTW
mtw = cfg.Analyzer(
    MTW,
    instance_label = 'mtw',
    met = 'met',
    electron = 'sel_iso_electrons',
    muon = 'sel_iso_muons'
)



from PhysicsTools.HeppyCore.analyzers.examples.ttbar.selection import Selection
selection = cfg.Analyzer(
    Selection,
    instance_label='cuts'
)

from PhysicsTools.HeppyCore.analyzers.examples.ttbar.TTbarTreeProducer import TTbarTreeProducer
gen_tree = cfg.Analyzer(
    TTbarTreeProducer,
    jets_30 = 'sel_jets_noelectronnomuon_30',
    m3 = 'm3',
    met = 'met',
    mtw= 'mtw',
    muons = 'sel_iso_muons',
    electrons = 'sel_iso_electrons'
)


# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence( [
    source,
    jets_30,
    muons,
    electrons,
    iso_muons,
    iso_electrons,
    match_jet_electrons,
    sel_jets_electron,
    match_muon_jets,
    sel_jets_muon,
    btagging,
    selection,
    m3, 
    mtw,
    gen_tree
    ] )

# comp.files.append('example_2.root')
#comp.splitFactor = len(comp.files)  # splitting the component in 2 chunks

config = cfg.Config(
    components = selectedComponents,
    sequence = sequence,
    services = [],
    events_class = Events
)

if __name__ == '__main__':
    import sys
    from PhysicsTools.HeppyCore.framework.looper import Looper

    def next():
        loop.process(loop.iEvent+1)

    loop = Looper( 'looper', config,
                   nEvents=100,
                   nPrint=0,
                   timeReport=True)
    loop.process(6)
    print loop.event
