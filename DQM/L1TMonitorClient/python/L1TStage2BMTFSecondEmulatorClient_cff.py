import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TdeStage2BMTF_cfi import ignoreBinsDeStage2Bmtf

# RegionalMuonCands
l1tStage2BMTFEmulatorSecondCompRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1TEMU/L1TdeStage2BMTF/L1TdeStage2BMTF-Secondary'),
    inputNum = cms.untracked.string('L1TEMU/L1TdeStage2BMTF/L1TdeStage2BMTF-Secondary/errorSummaryNum'),
    inputDen = cms.untracked.string('L1TEMU/L1TdeStage2BMTF/L1TdeStage2BMTF-Secondary/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between BMTF2 muons and BMTF2 emulator muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True),
    ignoreBin = cms.untracked.vint32(ignoreBinsDeStage2Bmtf)
)

# sequences
l1tStage2BMTFEmulatorSecondClient = cms.Sequence(
    l1tStage2BMTFEmulatorSecondCompRatioClient
)

