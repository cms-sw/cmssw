import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

# RegionalMuonCands
l1tStage2BMTFEmulatorCompRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1TEMU/L1TdeStage2BMTF'),
    inputNum = cms.untracked.string('L1TEMU/L1TdeStage2BMTF/errorSummaryNum'),
    inputDen = cms.untracked.string('L1TEMU/L1TdeStage2BMTF/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between BMTF muons and BMTF emulator muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# sequences
l1tStage2BMTFEmulatorClient = cms.Sequence(
    l1tStage2BMTFEmulatorCompRatioClient
)

