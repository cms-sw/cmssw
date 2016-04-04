import FWCore.ParameterSet.Config as cms

l1tStage2uGMT = cms.EDAnalyzer(
    "L1TStage2uGMT",
    ugmtSource = cms.InputTag("gmtStage2Digis", "Muon"),
    monitorDir = cms.untracked.string("L1T2016/L1TStage2uGMT"),
    verbose = cms.untracked.bool(False),
)

