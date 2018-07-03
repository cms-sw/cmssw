import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2Bmtf = DQMEDAnalyzer(
    "L1TStage2BMTF",
    bmtfSource = cms.InputTag("bmtfDigis", "BMTF"),
#    bmtfSourceTwinMux1 = cms.InputTag("BMTFStage2Digis", "TheDigis"),
#    bmtfSourceTwinMux2 = cms.InputTag("BMTFStage2Digis", "PhiDigis"),
    monitorDir = cms.untracked.string("L1T/L1TStage2BMTF"),
    verbose = cms.untracked.bool(False)
)

