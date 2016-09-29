import FWCore.ParameterSet.Config as cms

l1tdeStage2Emtf = cms.EDAnalyzer(
    "L1TdeStage2EMTF",
    dataSource = cms.InputTag("emtfStage2Digis"),
    emulSource = cms.InputTag("valEmtfStage2Digis", "EMTF"),
    monitorDir = cms.untracked.string("L1T2016EMU/L1TdeStage2EMTF"),
    verbose = cms.untracked.bool(False),
)

