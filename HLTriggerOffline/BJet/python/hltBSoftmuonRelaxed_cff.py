import FWCore.ParameterSet.Config as cms

import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
# HLT BTag Soft muon-based relaxed triggers analyers
hlt_BTagMu_DoubleJet60_Relaxed = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagMu_TripleJet40_Relaxed = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagMu_QuadJet30_Relaxed = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagMu_HT250_Relaxed = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
hltBSoftmuonRelaxed_modules = cms.PSet(
    modules = cms.vstring('hlt_BTagMu_DoubleJet60_Relaxed', 
        'hlt_BTagMu_TripleJet40_Relaxed', 
        'hlt_BTagMu_QuadJet30_Relaxed', 
        'hlt_BTagMu_HT250_Relaxed')
)
hltBSoftmuonRelaxed = cms.Sequence(hlt_BTagMu_DoubleJet60_Relaxed+hlt_BTagMu_TripleJet40_Relaxed+hlt_BTagMu_QuadJet30_Relaxed+hlt_BTagMu_HT250_Relaxed)
hlt_BTagMu_DoubleJet60_Relaxed.triggerPath = 'HLT_BTagMu_DoubleJet60_Relaxed'
hlt_BTagMu_DoubleJet60_Relaxed.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonNjetL1seeds","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon2jetL2filter60","","HLT"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter","","HLT"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBSoftmuonHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagMu_TripleJet40_Relaxed.triggerPath = 'HLT_BTagMu_TripleJet40_Relaxed'
hlt_BTagMu_TripleJet40_Relaxed.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonNjetL1seeds","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon3jetL2filter40","","HLT"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter","","HLT"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBSoftmuonHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagMu_QuadJet30_Relaxed.triggerPath = 'HLT_BTagMu_QuadJet30_Relaxed'
hlt_BTagMu_QuadJet30_Relaxed.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonNjetL1seeds","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon4jetL2filter30","","HLT"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter","","HLT"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBSoftmuonHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagMu_HT250_Relaxed.triggerPath = 'HLT_BTagMu_HT250_Relaxed'
hlt_BTagMu_HT250_Relaxed.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBSoftmuonHTL1seedsLowEnergy","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonHTL2filter250","","HLT"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL25filter","","HLT"),
        jets = cms.InputTag("hltBSoftmuonL25Jets","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonL3filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBSoftmuonHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))

