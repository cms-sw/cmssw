import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TdeStage2uGMT_cff import ignoreBins

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

l1tStage2uGTEmulatorCompRatioClientBXP1 = l1tStage2uGTEmulatorCompRatioClientBX0.clone()
l1tStage2uGTEmulatorCompRatioClientBXP2 = l1tStage2uGTEmulatorCompRatioClientBX0.clone()
l1tStage2uGTEmulatorCompRatioClientBXM1 = l1tStage2uGTEmulatorCompRatioClientBX0.clone()
l1tStage2uGTEmulatorCompRatioClientBXM2 = l1tStage2uGTEmulatorCompRatioClientBX0.clone()


BX            = 'BX1'
errHistNumStr = 'dataEmulSummary_' + BX
errHistDenStr = 'normalizationHisto'
ratioHistStr  = 'dataEmulMismatchRatio_' + BX
l1tStage2uGTEmulatorCompRatioClientBXP1.inputNum  = cms.untracked.string(ugmtEmuDqmDir+'/'+errHistNumStr)
l1tStage2uGTEmulatorCompRatioClientBXP1.inputDen  = cms.untracked.string(ugmtEmuDqmDir+'/'+errHistDenStr)
l1tStage2uGTEmulatorCompRatioClientBXP1.ratioName = cms.untracked.string(ratioHistStr)

BX            = 'BX2'
errHistNumStr = 'dataEmulSummary_' + BX
errHistDenStr = 'normalizationHisto'
ratioHistStr  = 'dataEmulMismatchRatio_' + BX
l1tStage2uGTEmulatorCompRatioClientBXP2.inputNum  = cms.untracked.string(ugmtEmuDqmDir+'/'+errHistNumStr)
l1tStage2uGTEmulatorCompRatioClientBXP2.inputDen  = cms.untracked.string(ugmtEmuDqmDir+'/'+errHistDenStr)
l1tStage2uGTEmulatorCompRatioClientBXP2.ratioName = cms.untracked.string(ratioHistStr)

BX            = 'BX-1'
errHistNumStr = 'dataEmulSummary_' + BX
errHistDenStr = 'normalizationHisto'
ratioHistStr  = 'dataEmulMismatchRatio_' + BX
l1tStage2uGTEmulatorCompRatioClientBXM1.inputNum  = cms.untracked.string(ugmtEmuDqmDir+'/'+errHistNumStr)
l1tStage2uGTEmulatorCompRatioClientBXM1.inputDen  = cms.untracked.string(ugmtEmuDqmDir+'/'+errHistDenStr)
l1tStage2uGTEmulatorCompRatioClientBXM1.ratioName = cms.untracked.string(ratioHistStr)

BX            = 'BX-2'
errHistNumStr = 'dataEmulSummary_' + BX
errHistDenStr = 'normalizationHisto'
ratioHistStr  = 'dataEmulMismatchRatio_' + BX
l1tStage2uGTEmulatorCompRatioClientBXM2.inputNum  = cms.untracked.string(ugmtEmuDqmDir+'/'+errHistNumStr)
l1tStage2uGTEmulatorCompRatioClientBXM2.inputDen  = cms.untracked.string(ugmtEmuDqmDir+'/'+errHistDenStr)
l1tStage2uGTEmulatorCompRatioClientBXM2.ratioName = cms.untracked.string(ratioHistStr)

# uGT

# sequences
l1tStage2uGTEmulatorClient = cms.Sequence(
    l1tStage2uGTEmulatorCompRatioClientBX0 +
    l1tStage2uGTEmulatorCompRatioClientBXP1 +
    l1tStage2uGTEmulatorCompRatioClientBXP2 +
    l1tStage2uGTEmulatorCompRatioClientBXM1 +
    l1tStage2uGTEmulatorCompRatioClientBXM2
)

