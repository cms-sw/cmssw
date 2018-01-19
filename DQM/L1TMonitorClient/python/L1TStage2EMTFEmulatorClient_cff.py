import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TdeStage2EMTF_cff import ignoreBinsDeStage2Emtf

# RegionalMuonCands
l1tStage2EMTFEmulatorCompRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string('L1TEMU/L1TdeStage2EMTF'),
    inputNum = cms.untracked.string('L1TEMU/L1TdeStage2EMTF/errorSummaryNum'),
    inputDen = cms.untracked.string('L1TEMU/L1TdeStage2EMTF/errorSummaryDen'),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between EMTF muons and EMTF emulator muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True),
    ignoreBin = cms.untracked.vint32(ignoreBinsDeStage2Emtf)
)

# sequences
l1tStage2EMTFEmulatorClient = cms.Sequence(
    l1tStage2EMTFEmulatorCompRatioClient
)

