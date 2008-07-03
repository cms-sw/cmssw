import FWCore.ParameterSet.Config as cms

import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
# HLT Btag Softmuon paths analyers
hltb1jetmu = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hltb2jetmu = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hltb3jetmu = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hltb4jetmu = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hltbhtmu = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
hltBSoftmuon_modules = cms.PSet(
    modules = cms.vstring('hltb1jetmu', 
        'hltb2jetmu', 
        'hltb3jetmu', 
        'hltb4jetmu', 
        'hltbhtmu')
)
hltBSoftmuon = cms.Sequence(hltb1jetmu+hltb2jetmu+hltb3jetmu+hltb4jetmu+hltbhtmu)
hltb1jetmu.triggerPath = 'HLTB1JetMu'
hltb1jetmu.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonNjetL1seeds","","HLT"),
    jets = cms.InputTag("none"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon1jetL2filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonByDRL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb2jetmu.triggerPath = 'HLTB2JetMu'
hltb2jetmu.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonNjetL1seeds","","HLT"),
    jets = cms.InputTag("none"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon2jetL2filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb3jetmu.triggerPath = 'HLTB3JetMu'
hltb3jetmu.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonNjetL1seeds","","HLT"),
    jets = cms.InputTag("none"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon3jetL2filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltb4jetmu.triggerPath = 'HLTB4JetMu'
hltb4jetmu.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonNjetL1seeds","","HLT"),
    jets = cms.InputTag("none"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon4jetL2filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hltbhtmu.triggerPath = 'HLTBHTMu'
hltbhtmu.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonHTL1seeds","","HLT"),
    jets = cms.InputTag("none"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonHTL2filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))

