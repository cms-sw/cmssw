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

from DQM.L1TMonitorClient.L1TdeStage2RegionalShowerClient_cfi import *

# sequences
l1tStage2EMTFEmulatorClient = cms.Sequence(
    l1tStage2EMTFEmulatorCompRatioClient
)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
_run3shower_l1tStage2EMTFEmulatorClient = l1tStage2EMTFEmulatorClient.copy()
run3_GEM.toReplaceWith(l1tStage2EMTFEmulatorClient, cms.Sequence(_run3shower_l1tStage2EMTFEmulatorClient + l1tdeStage2RegionalShowerClient))
