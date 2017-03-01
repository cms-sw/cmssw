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
    egToken    = cms.InputTag("gtStage2Digis","EGamma"),
    tauTokens  = cms.VInputTag(cms.InputTag("gtStage2Digis","Tau")),
    jetToken   = cms.InputTag("gtStage2Digis","Jet"),
    sumToken   = cms.InputTag("gtStage2Digis","EtSum"),
    muonToken  = cms.InputTag("gtStage2Digis","Muon"),
)
