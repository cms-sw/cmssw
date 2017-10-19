import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
# from DQM.L1TMonitor.L1TStage2uGMT_cff import ignoreBins

# directory path shortening
ugtDqmDir = 'L1T/L1TStage2uGT'
calol2uGTDir = 'calol2ouput_vs_uGTinput'
# ugmtMuCpyDqmDir = ugmtDqmDir+'/uGMTMuonCopies'
# ugmtZSDqmDir = ugmtDqmDir+'/zeroSuppression'
# input histograms
errHistNumStr = 'errorSummaryNum'
errHistDenStr = 'errorSummaryDen'

# Muons
l1tStage2uGTvsCaloLayer2RatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugtDqmDir + '/' + calol2uGTDir),
    inputNum = cms.untracked.string(ugtDqmDir + '/' + calol2uGTDir + '/' + errHistNumStr),
    inputDen = cms.untracked.string(ugtDqmDir + '/' + calol2uGTDir + '/'+ errHistDenStr),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between CaloLayer2 outputs and uGT inputs'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# sequences
l1tStage2uGTCaloLayer2CompClient = cms.Sequence(
    l1tStage2uGTvsCaloLayer2RatioClient
)

