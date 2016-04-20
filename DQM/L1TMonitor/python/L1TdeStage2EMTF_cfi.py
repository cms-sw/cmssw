import FWCore.ParameterSet.Config as cms

l1tdeStage2Emtf = cms.EDAnalyzer(
    "L1TdeStage2EMTF",
    dataProducer = cms.InputTag("emtfStage2Digis"),
    emulProducer = cms.InputTag("valEmtfStage2Digis", "EMTF"),
    monitorDir = cms.untracked.string("L1TEMU2016/L1TdeStage2EMTF"),
    verbose = cms.untracked.bool(False),
)

