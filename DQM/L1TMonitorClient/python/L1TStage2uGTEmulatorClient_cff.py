import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

# directories
ugmtEmuDqmDir = "L1TEMU/L1TdeStage2uGT"


BX            = 'CentralBX'
errHistNumStr = 'dataEmulSummary_' + BX
errHistDenStr = 'normalizationHisto'
ratioHistStr  = 'dataEmulMismatchRatio_' + BX

l1tStage2uGTEmulatorCompRatioClientBX0 = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugmtEmuDqmDir),
    inputNum = cms.untracked.string(ugmtEmuDqmDir+'/'+errHistNumStr),
    inputDen = cms.untracked.string(ugmtEmuDqmDir+'/'+errHistDenStr),
    ratioName = cms.untracked.string(ratioHistStr),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGT emulator and data'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

BX            = 'BX1'
errHistNumStr = 'dataEmulSummary_' + BX
errHistDenStr = 'normalizationHisto'
ratioHistStr  = 'dataEmulMismatchRatio_' + BX
l1tStage2uGTEmulatorCompRatioClientBXP1 = l1tStage2uGTEmulatorCompRatioClientBX0.clone(
    inputNum  = ugmtEmuDqmDir+'/'+errHistNumStr,
    inputDen  = ugmtEmuDqmDir+'/'+errHistDenStr,
    ratioName = ratioHistStr
)

BX            = 'BX2'
errHistNumStr = 'dataEmulSummary_' + BX
errHistDenStr = 'normalizationHisto'
ratioHistStr  = 'dataEmulMismatchRatio_' + BX
l1tStage2uGTEmulatorCompRatioClientBXP2 = l1tStage2uGTEmulatorCompRatioClientBX0.clone(
    inputNum  = ugmtEmuDqmDir+'/'+errHistNumStr,
    inputDen  = ugmtEmuDqmDir+'/'+errHistDenStr,
    ratioName = ratioHistStr
)

BX            = 'BX-1'
errHistNumStr = 'dataEmulSummary_' + BX
errHistDenStr = 'normalizationHisto'
ratioHistStr  = 'dataEmulMismatchRatio_' + BX
l1tStage2uGTEmulatorCompRatioClientBXM1 = l1tStage2uGTEmulatorCompRatioClientBX0.clone(
    inputNum  = ugmtEmuDqmDir+'/'+errHistNumStr,
    inputDen  = ugmtEmuDqmDir+'/'+errHistDenStr,
    ratioName = ratioHistStr
)

BX            = 'BX-2'
errHistNumStr = 'dataEmulSummary_' + BX
errHistDenStr = 'normalizationHisto'
ratioHistStr  = 'dataEmulMismatchRatio_' + BX
l1tStage2uGTEmulatorCompRatioClientBXM2 = l1tStage2uGTEmulatorCompRatioClientBX0.clone(
    inputNum  = ugmtEmuDqmDir+'/'+errHistNumStr,
    inputDen  = ugmtEmuDqmDir+'/'+errHistDenStr,
    ratioName = ratioHistStr
)

# uGT

# sequences
l1tStage2uGTEmulatorClient = cms.Sequence(
    l1tStage2uGTEmulatorCompRatioClientBX0 +
    l1tStage2uGTEmulatorCompRatioClientBXP1 +
    l1tStage2uGTEmulatorCompRatioClientBXP2 +
    l1tStage2uGTEmulatorCompRatioClientBXM1 +
    l1tStage2uGTEmulatorCompRatioClientBXM2
)

