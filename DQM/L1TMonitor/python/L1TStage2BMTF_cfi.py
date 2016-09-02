import FWCore.ParameterSet.Config as cms

l1tStage2Bmtf = cms.EDAnalyzer(
    "L1TStage2BMTF",
    bmtfSource = cms.InputTag("bmtfDigis", "BMTF"),
#    bmtfSourceTwinMux1 = cms.InputTag("BMTFStage2Digis", "TheDigis"),
#    bmtfSourceTwinMux2 = cms.InputTag("BMTFStage2Digis", "PhiDigis"),
    monitorDir = cms.untracked.string("L1T2016/L1TStage2BMTF"),
    verbose = cms.untracked.bool(False),
)

