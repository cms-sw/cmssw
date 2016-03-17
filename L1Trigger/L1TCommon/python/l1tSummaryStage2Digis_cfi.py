import FWCore.ParameterSet.Config as cms

l1tSummaryStage2Digis = cms.EDAnalyzer(
    "L1TSummary",
    tag        = cms.string("Stage 2 Unpacked Digis"),
    egCheck    = cms.bool(True),
    tauCheck   = cms.bool(True),
    jetCheck   = cms.bool(True),
    sumCheck   = cms.bool(True),
    muonCheck  = cms.bool(True),
    bxZeroOnly = cms.bool(True),
    egToken    = cms.InputTag("caloStage2Digis","EGamma"),
    tauTokens  = cms.VInputTag(cms.InputTag("caloStage2Digis","Tau")),
    jetToken   = cms.InputTag("caloStage2Digis","Jet"),
    sumToken   = cms.InputTag("caloStage2Digis","EtSum"),
    muonToken  = cms.InputTag("gmtStage2Digis","Muon"),
)
