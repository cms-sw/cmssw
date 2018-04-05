import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeStage2Emtf = DQMEDAnalyzer(
    "L1TdeStage2EMTF",
    dataSource = cms.InputTag("emtfStage2Digis"),
    emulSource = cms.InputTag("valEmtfStage2Digis", "EMTF"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2EMTF"),
    verbose = cms.untracked.bool(False),
)

