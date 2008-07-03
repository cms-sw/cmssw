import FWCore.ParameterSet.Config as cms

import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
# HLT BTag IP-based triggers analyers
hlt_BTagIP_Jet180 = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagIP_DoubleJet120 = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagIP_TripleJet70 = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagIP_QuadJet40 = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagIP_HT470 = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
hltBLifetime_modules = cms.PSet(
    modules = cms.vstring('hlt_BTagIP_Jet180', 
        'hlt_BTagIP_DoubleJet120', 
        'hlt_BTagIP_TripleJet70', 
        'hlt_BTagIP_QuadJet40', 
        'hlt_BTagIP_HT470')
)
hltBLifetime = cms.Sequence(hlt_BTagIP_Jet180+hlt_BTagIP_DoubleJet120+hlt_BTagIP_TripleJet70+hlt_BTagIP_QuadJet40+hlt_BTagIP_HT470)
hlt_BTagIP_Jet180.triggerPath = 'HLT_BTagIP_Jet180'
hlt_BTagIP_Jet180.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime1jetL2filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeL25Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25Associator","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeL3Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3Associator","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagIP_DoubleJet120.triggerPath = 'HLT_BTagIP_DoubleJet120'
hlt_BTagIP_DoubleJet120.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime2jetL2filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeL25Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25Associator","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeL3Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3Associator","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagIP_TripleJet70.triggerPath = 'HLT_BTagIP_TripleJet70'
hlt_BTagIP_TripleJet70.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime3jetL2filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeL25Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25Associator","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeL3Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3Associator","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagIP_QuadJet40.triggerPath = 'HLT_BTagIP_QuadJet40'
hlt_BTagIP_QuadJet40.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetime4jetL2filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeL25Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25Associator","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeL3Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3Associator","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagIP_HT470.triggerPath = 'HLT_BTagIP_HT470'
hlt_BTagIP_HT470.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltBLifetimeL1seeds","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeHTL2filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeL25Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL25Associator","","HLT"),
        name = cms.string('L2'),
        title = cms.string('L2')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL25filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeL3Jets","","HLT"),
        tracks = cms.InputTag("hltBLifetimeL3Associator","","HLT"),
        name = cms.string('L25'),
        title = cms.string('L2.5')
    ), 
    cms.PSet(
        filter = cms.InputTag("hltBLifetimeL3filter","","HLT"),
        jets = cms.InputTag("hltBLifetimeHLTJets"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))

