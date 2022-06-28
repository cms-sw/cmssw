import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeStage2Cppf = DQMEDAnalyzer(
    "L1TdeStage2CPPF",
    dataSource = cms.InputTag("rpcCPPFRawToDigi"),
    emulSource = cms.InputTag("valCppfStage2Digis","recHit"),
    monitorDir = cms.untracked.string("L1TEMU/L1TdeStage2CPPF"),
    verbose = cms.untracked.bool(False),
)

