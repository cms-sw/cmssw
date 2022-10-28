import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

l1tdeCSCTPGShower = DQMEDAnalyzer(
    "L1TdeCSCTPGShower",
    # CSC Showers not in data yet. Use Emul for both
    # Once Run 3 firmware are implemented, should change data tags to 
    # cms.InputTag("muonCSCDigis", "MuonCSCShowerDigi") for correlated shower,
    # something like cms.InputTag("muonCSCDigis", "MuonCSCShowerDigiAnode") for Anode shower,
    # and something like cms.InputTag("muonCSCDigis", "MuonCSCShowerDigiCathode") for Cathode shower
    # - 2021.12.06 Xunwu Zuo
    dataALCTShower = cms.InputTag("muonCSCDigis", "MuonCSCShowerDigiAnode"),
    emulALCTShower = cms.InputTag("valCscStage2Digis", "Anode"),
    dataCLCTShower = cms.InputTag("muonCSCDigis", "MuonCSCShowerDigiCathode"),
    emulCLCTShower = cms.InputTag("valCscStage2Digis", "Cathode"),
    dataLCTShower = cms.InputTag("muonCSCDigis","MuonCSCShowerDigi"),
    emulLCTShower = cms.InputTag("valCscStage2Digis"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeCSCTPGShower"),
)

