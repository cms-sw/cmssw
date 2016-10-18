import FWCore.ParameterSet.Config as cms

l1tEfficiencyMuons_offline = cms.EDAnalyzer("L1TEfficiencyMuons_Offline",
    verbose   = cms.untracked.bool(False),
    gmtPtCuts = cms.untracked.vint32(16,20,25),
    muonInputTag = cms.untracked.InputTag("muons"),
    gmtInputTag  = cms.untracked.InputTag("gmtStage2Digis"),
    vtxInputTag = cms.untracked.InputTag("offlinePrimaryVertices"),
    bsInputTag  = cms.untracked.InputTag("offlineBeamSpot"),

    triggerNames = cms.untracked.vstring(
        "HLT_IsoMu18_v*",
        "HLT_IsoMu20_v*",
        "HLT_IsoMu22_v*",
        "HLT_IsoMu24_v*",
        "HLT_IsoMu27_v*",
        "HLT_Mu30_v*",
        "HLT_Mu40_v*"
    ),
    trigInputTag       = cms.untracked.InputTag("hltTriggerSummaryAOD", "", "HLT"),
    trigProcess        = cms.untracked.string("HLT"),
    trigProcess_token  = cms.untracked.InputTag("TriggerResults","","HLT")
)
