import FWCore.ParameterSet.Config as cms

import copy
from HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi import *
# HLT Btag Lifetime paths analyers
hltb1jet = copy.deepcopy(hltBtagLifetimeAnalyzer)
import copy
from HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi import *
hltb2jet = copy.deepcopy(hltBtagLifetimeAnalyzer)
import copy
from HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi import *
hltb3jet = copy.deepcopy(hltBtagLifetimeAnalyzer)
import copy
from HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi import *
hltb4jet = copy.deepcopy(hltBtagLifetimeAnalyzer)
import copy
from HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi import *
hltbht = copy.deepcopy(hltBtagLifetimeAnalyzer)
hltBLifetimeModules = cms.PSet(
    modules = cms.vstring('hltb1jet', 
        'hltb2jet', 
        'hltb3jet', 
        'hltb4jet', 
        'hltbht')
)
hltBLifetime = cms.Sequence(hltb1jet+hltb2jet+hltb3jet+hltb4jet+hltbht)
hltb1jet.triggerPath = 'HLTB1Jet'
hltb1jet.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds"),
    jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
    name = cms.string('preL2'),
    title = cms.string('pre-L2')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime1jetL2filter"),
        jets = cms.InputTag("hltBLifetimeL25Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25Associator","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter"),
        jets = cms.InputTag("hltBLifetimeL3Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3Associator","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter"),
        jets = cms.InputTag("hltBLifetimeHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb2jet.triggerPath = 'HLTB2Jet'
hltb2jet.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds"),
    jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
    name = cms.string('preL2'),
    title = cms.string('pre-L2')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime2jetL2filter"),
        jets = cms.InputTag("hltBLifetimeL25Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25Associator","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter"),
        jets = cms.InputTag("hltBLifetimeL3Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3Associator","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter"),
        jets = cms.InputTag("hltBLifetimeHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb3jet.triggerPath = 'HLTB3Jet'
hltb3jet.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds"),
    jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
    name = cms.string('preL2'),
    title = cms.string('pre-L2')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime3jetL2filter"),
        jets = cms.InputTag("hltBLifetimeL25Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25Associator","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter"),
        jets = cms.InputTag("hltBLifetimeL3Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3Associator","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter"),
        jets = cms.InputTag("hltBLifetimeHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb4jet.triggerPath = 'HLTB4Jet'
hltb4jet.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds"),
    jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
    name = cms.string('preL2'),
    title = cms.string('pre-L2')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime4jetL2filter"),
        jets = cms.InputTag("hltBLifetimeL25Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25Associator","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter"),
        jets = cms.InputTag("hltBLifetimeL3Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3Associator","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter"),
        jets = cms.InputTag("hltBLifetimeHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltbht.triggerPath = 'HLTBHT'
hltbht.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds"),
    jets = cms.InputTag("iterativeCone5CaloJets","","HLT"),
    name = cms.string('preL2'),
    title = cms.string('pre-L2')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeHTL2filter"),
        jets = cms.InputTag("hltBLifetimeL25Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25Associator","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter"),
        jets = cms.InputTag("hltBLifetimeL3Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3Associator","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter"),
        jets = cms.InputTag("hltBLifetimeHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))

