import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeStage2RegionalShower = DQMEDAnalyzer(
    "L1TdeStage2RegionalShower",
    dataSource = cms.InputTag("emtfStage2Digis"),
    emulSource = cms.InputTag("valEmtfStage2Showers", "EMTF"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2EMTF/Shower"),
)

