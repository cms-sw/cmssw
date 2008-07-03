import FWCore.ParameterSet.Config as cms

import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
# HLT Btag Lifetime paths analyers
hltb1jet = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hltb2jet = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hltb3jet = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hltb4jet = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hltbht = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
hltBLifetime_modules = cms.PSet(
    modules = cms.vstring('hltb1jet', 
        'hltb2jet', 
        'hltb3jet', 
        'hltb4jet', 
        'hltbht')
)
hltBLifetime = cms.Sequence(hltb1jet+hltb2jet+hltb3jet+hltb4jet+hltbht)
hltb1jet.triggerPath = 'HLTB1Jet'
hltb1jet.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds","","HLT"),
    jets = cms.InputTag("none"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime1jetL2filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb2jet.triggerPath = 'HLTB2Jet'
hltb2jet.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds","","HLT"),
    jets = cms.InputTag("none"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime2jetL2filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb3jet.triggerPath = 'HLTB3Jet'
hltb3jet.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds","","HLT"),
    jets = cms.InputTag("none"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime3jetL2filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb4jet.triggerPath = 'HLTB4Jet'
hltb4jet.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds","","HLT"),
    jets = cms.InputTag("none"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime4jetL2filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltbht.triggerPath = 'HLTBHT'
hltbht.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds","","HLT"),
    jets = cms.InputTag("none"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeHTL2filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
