import FWCore.ParameterSet.Config as cms

l1tStage2BMTF = cms.EDAnalyzer(
    "L1TStage2BMTF",
    bmtfSource = cms.InputTag("BMTFStage2Digis"),
    monitorDir = cms.untracked.string("L1T2016/L1TStage2BMTF"),
    verbose = cms.untracked.bool(False),
)

