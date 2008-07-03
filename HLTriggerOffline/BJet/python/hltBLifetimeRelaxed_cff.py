import FWCore.ParameterSet.Config as cms

import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
# HLT BTag IP-based relaxed triggers analyers
hlt_BTagIP_Jet120_Relaxed = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagIP_DoubleJet60_Relaxed = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagIP_TripleJet40_Relaxed = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagIP_QuadJet30_Relaxe = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagIP_HT320_Relaxed = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
hltBLifetimeRelaxed_modules = cms.PSet(
    modules = cms.vstring('hlt_BTagIP_Jet120_Relaxed', 
        'hlt_BTagIP_DoubleJet60_Relaxed', 
        'hlt_BTagIP_TripleJet40_Relaxed', 
        'hlt_BTagIP_QuadJet30_Relaxe', 
        'hlt_BTagIP_HT320_Relaxed')
)
hltBLifetimeRelaxed = cms.Sequence(hlt_BTagIP_Jet120_Relaxed+hlt_BTagIP_DoubleJet60_Relaxed+hlt_BTagIP_TripleJet40_Relaxed+hlt_BTagIP_QuadJet30_Relaxe+hlt_BTagIP_HT320_Relaxed)
hlt_BTagIP_Jet120_Relaxed.triggerPath = 'HLT_BTagIP_Jet120_Relaxed'
hlt_BTagIP_Jet120_Relaxed.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seedsLowEnergy","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime1jetL2filter120","","HLT"),
        jets = cms.InputTag("hltBLifetimeL25JetsRelaxed","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25AssociatorRelaxed","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBLifetimeL3JetsRelaxed","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3AssociatorRelaxed","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBLifetimeHLTJetsRelaxed"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagIP_DoubleJet60_Relaxed.triggerPath = 'HLT_BTagIP_DoubleJet60_Relaxed'
hlt_BTagIP_DoubleJet60_Relaxed.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seedsLowEnergy","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime2jetL2filter60","","HLT"),
        jets = cms.InputTag("hltBLifetimeL25JetsRelaxed","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25AssociatorRelaxed","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBLifetimeL3JetsRelaxed","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3AssociatorRelaxed","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBLifetimeHLTJetsRelaxed"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagIP_TripleJet40_Relaxed.triggerPath = 'HLT_BTagIP_TripleJet40_Relaxed'
hlt_BTagIP_TripleJet40_Relaxed.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seedsLowEnergy","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime3jetL2filter40","","HLT"),
        jets = cms.InputTag("hltBLifetimeL25JetsRelaxed","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25AssociatorRelaxed","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBLifetimeL3JetsRelaxed","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3AssociatorRelaxed","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBLifetimeHLTJetsRelaxed"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagIP_QuadJet30_Relaxe.triggerPath = 'HLT_BTagIP_QuadJet30_Relaxed'
hlt_BTagIP_QuadJet30_Relaxe.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seedsLowEnergy","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime4jetL2filter30","","HLT"),
        jets = cms.InputTag("hltBLifetimeL25JetsRelaxed","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25AssociatorRelaxed","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBLifetimeL3JetsRelaxed","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3AssociatorRelaxed","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBLifetimeHLTJetsRelaxed"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagIP_HT320_Relaxed.triggerPath = 'HLT_BTagIP_HT320_Relaxed'
hlt_BTagIP_HT320_Relaxed.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seedsLowEnergy","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeHTL2filter320","","HLT"),
        jets = cms.InputTag("hltBLifetimeL25JetsRelaxed","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25AssociatorRelaxed","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBLifetimeL3JetsRelaxed","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3AssociatorRelaxed","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filterRelaxed","","HLT"),
        jets = cms.InputTag("hltBLifetimeHLTJetsRelaxed"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))

