import FWCore.ParameterSet.Config as cms

l1tStage2Omtf = cms.EDAnalyzer(
    "L1TStage2OMTF",
    omtfSource = cms.InputTag("omtfStage2Digis", ""),
    monitorDir = cms.untracked.string("L1T/L1TStage2OMTF"),
    verbose = cms.untracked.bool(False),
)

