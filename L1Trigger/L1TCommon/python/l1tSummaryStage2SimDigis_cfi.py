import FWCore.ParameterSet.Config as cms

l1tSummaryStage2SimDigis = cms.EDAnalyzer(
    "L1TSummary",
    tag        = cms.string("Stage 2 Simulated Digis"),
    egCheck    = cms.bool(True),
    tauCheck   = cms.bool(True),
    jetCheck   = cms.bool(True),
    sumCheck   = cms.bool(True),
    muonCheck  = cms.bool(True),
    bxZeroOnly = cms.bool(True),
    egToken    = cms.InputTag("simCaloStage2Digis"),
    tauTokens  = cms.VInputTag("simCaloStage2Digis"),
    jetToken   = cms.InputTag("simCaloStage2Digis"),
    sumToken   = cms.InputTag("simCaloStage2Digis"),
    muonToken  = cms.InputTag("simGmtStage2Digis",""),
)
