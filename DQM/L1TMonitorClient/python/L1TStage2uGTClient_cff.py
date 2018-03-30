import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TStage2uGT_cff import l1tStage2uGMTOutVsuGTIn

# directory path shortening
ugtDqmDir = 'L1T/L1TStage2uGT'
# input histograms
errHistNumStr = 'errorSummaryNum'
errHistDenStr = 'errorSummaryDen'

# Muons
l1tStage2uGMTOutVsuGTInRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput'),
    inputNum = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput/'+errHistNumStr),
    inputDen = cms.untracked.string(ugtDqmDir+'/uGMToutput_vs_uGTinput/'+errHistDenStr),
    ignoreBin = cms.untracked.vint32(l1tStage2uGMTOutVsuGTIn.ignoreBin),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT output muons and uGT input muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# sequences
l1tStage2uGTClient = cms.Sequence(
    l1tStage2uGMTOutVsuGTInRatioClient
)
