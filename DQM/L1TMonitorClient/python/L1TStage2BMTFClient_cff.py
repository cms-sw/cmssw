import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

# directory path shortening
bmtfDqmDir = 'L1T/L1TStage2BMTF'
bmtfZSDqmDir = bmtfDqmDir+'/zeroSuppression'
errHistNumStr = 'errorSummaryNum'
errHistDenStr = 'errorSummaryDen'

# zero suppression
l1tStage2BmtfZeroSuppRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(bmtfZSDqmDir+'/AllEvts'),
    inputNum = cms.untracked.string(bmtfZSDqmDir+'/AllEvts/'+errHistNumStr),
    inputDen = cms.untracked.string(bmtfZSDqmDir+'/AllEvts/'+errHistDenStr),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of bad zero suppression rates'),
    yAxisTitle = cms.untracked.string('# fail / # total'),
    binomialErr = cms.untracked.bool(True)
)

l1tStage2BmtfZeroSuppFatEvtsRatioClient = l1tStage2BmtfZeroSuppRatioClient.clone(
    monitorDir = bmtfZSDqmDir+'/FatEvts',
    inputNum = bmtfZSDqmDir+'/FatEvts/'+errHistNumStr,
    inputDen = bmtfZSDqmDir+'/FatEvts/'+errHistDenStr,
    ratioTitle = 'Summary of bad zero suppression rates'
)
# sequences
l1tStage2BmtfZeroSuppCompClient = cms.Sequence(
    l1tStage2BmtfZeroSuppRatioClient
  + l1tStage2BmtfZeroSuppFatEvtsRatioClient
)

l1tStage2BmtfClient = cms.Sequence(
    l1tStage2BmtfZeroSuppCompClient
)

