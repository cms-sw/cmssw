import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TdeStage2BMTF_cfi import ignoreBinsDeStage2Bmtf

# RegionalMuonCands
l1tStage2kBMTFEmulatorCompRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1TEMU/L1TdeStage2BMTF/L1TdeStage2KalmanBMTF'),
    inputNum = cms.untracked.string('L1TEMU/L1TdeStage2BMTF/L1TdeStage2KalmanBMTF/errorSummaryNum'),
    inputDen = cms.untracked.string('L1TEMU/L1TdeStage2BMTF/L1TdeStage2KalmanBMTF/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between Kalman BMTF muons and Kalman BMTF emulator muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True),
    ignoreBin = cms.untracked.vint32(ignoreBinsDeStage2Bmtf)
)

# sequences
l1tStage2kBMTFEmulatorClient = cms.Sequence(
    l1tStage2kBMTFEmulatorCompRatioClient
)

