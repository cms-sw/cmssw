import FWCore.ParameterSet.Config as cms

# HLT BTag Soft muon-based triggers analyers
import HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi
hlt_BTagMu_Jet20_Calib  = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
hlt_BTagMu_DoubleJet120 = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
hlt_BTagMu_TripleJet70  = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
hlt_BTagMu_QuadJet40    = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()
hlt_BTagMu_HT370        = HLTriggerOffline.BJet.hltBtagLifetimeAnalyzer_cfi.hltBtagLifetimeAnalyzer.clone()

hltBSoftmuon_modules = cms.PSet(
    modules = cms.vstring('hlt_BTagMu_Jet20_Calib', 
        'hlt_BTagMu_DoubleJet120', 
        'hlt_BTagMu_TripleJet70', 
        'hlt_BTagMu_QuadJet40', 
        'hlt_BTagMu_HT370')
)
hltBSoftmuon = cms.Sequence(hlt_BTagMu_Jet20_Calib+hlt_BTagMu_DoubleJet120+hlt_BTagMu_TripleJet70+hlt_BTagMu_QuadJet40+hlt_BTagMu_HT370)
hlt_BTagMu_Jet20_Calib.triggerPath = 'HLT_BTagMu_Jet20_Calib'
hlt_BTagMu_Jet20_Calib.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltL1sBSoftmuonNjet","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon1jetL2filter","","HLT"),
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
        filter = cms.InputTag("hltBSoftmuonByDRL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagMu_DoubleJet120.triggerPath = 'HLT_BTagMu_DoubleJet120'
hlt_BTagMu_DoubleJet120.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltL1sBSoftmuonNjet","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon2jetL2filter","","HLT"),
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
        filter = cms.InputTag("hltBSoftmuonL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagMu_TripleJet70.triggerPath = 'HLT_BTagMu_TripleJet70'
hlt_BTagMu_TripleJet70.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltL1sBSoftmuonNjet","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon3jetL2filter","","HLT"),
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
        filter = cms.InputTag("hltBSoftmuonL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagMu_QuadJet40.triggerPath = 'HLT_BTagMu_QuadJet40'
hlt_BTagMu_QuadJet40.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltL1sBSoftmuonNjet","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuon4jetL2filter","","HLT"),
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
        filter = cms.InputTag("hltBSoftmuonL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))
hlt_BTagMu_HT370.triggerPath = 'HLT_BTagMu_HT370'
hlt_BTagMu_HT370.levels = cms.VPSet(cms.PSet(
    filter = cms.InputTag("hltL1sBSoftMuonHT","","HLT"),
    jets = cms.InputTag("hltIterativeCone5CaloJets","","HLT"),
    name = cms.string('L1'),
    title = cms.string('L1')
), 
    cms.PSet(
        filter = cms.InputTag("hltBSoftmuonHTL2filter","","HLT"),
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
        filter = cms.InputTag("hltBSoftmuonL3filter","","HLT"),
        jets = cms.InputTag("none"),
        name = cms.string('L3'),
        title = cms.string('L3')
    ))

