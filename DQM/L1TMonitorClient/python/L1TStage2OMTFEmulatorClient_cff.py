import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TdeStage2OMTF_cfi import ignoreBins

# RegionalMuonCands
l1tStage2OMTFEmulatorCompRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1TEMU/L1TdeStage2OMTF'),
    inputNum = cms.untracked.string('L1TEMU/L1TdeStage2OMTF/errorSummaryNum'),
    inputDen = cms.untracked.string('L1TEMU/L1TdeStage2OMTF/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between OMTF muons and OMTF emulator muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True),
    ignoreBin = cms.untracked.vint32(ignoreBins)
)

# sequences
l1tStage2OMTFEmulatorClient = cms.Sequence(
    l1tStage2OMTFEmulatorCompRatioClient
)

