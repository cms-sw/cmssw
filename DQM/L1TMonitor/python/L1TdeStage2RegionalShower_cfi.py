import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeStage2RegionalShower = DQMEDAnalyzer(
    "L1TdeStage2RegionalShower",
    # EMTF Showers not in data yet. Use Emul for both
    # Once Run 3 firmware are implemented, should change data tags to 
    # cms.InputTag("emtfStage2Digis")
    # - 2021.12.06 Xunwu Zuo

#    dataSource = cms.InputTag("simEmtfShowers", "EMTF"),
#    emulSource = cms.InputTag("simEmtfShowers", "EMTF"),
    dataSource = cms.InputTag("valEmtfStage2Showers", "EMTF"),
    emulSource = cms.InputTag("valEmtfStage2Showers", "EMTF"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2EMTF/Shower"),
)

