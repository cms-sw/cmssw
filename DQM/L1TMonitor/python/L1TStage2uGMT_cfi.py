import FWCore.ParameterSet.Config as cms

# the uGMT DQM module
l1tStage2uGMT = cms.EDAnalyzer(
    "L1TStage2uGMT",
    bmtfProducer = cms.InputTag("gmtStage2Digis", "BMTF"),
    omtfProducer = cms.InputTag("gmtStage2Digis", "OMTF"),
    emtfProducer = cms.InputTag("gmtStage2Digis", "EMTF"),
    muonProducer = cms.InputTag("gmtStage2Digis", "Muon"),
    monitorDir = cms.untracked.string("L1T/L1TStage2uGMT"),
    emulator = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False),
)

