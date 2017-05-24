import FWCore.ParameterSet.Config as cms

# directories
ugmtEmuDqmDir = "L1TEMU/L1TdeStage2uGMT"
ugmtEmuDEDqmDir = ugmtEmuDqmDir+"/data_vs_emulator_comparison"
ugmtEmuImdMuDqmDir = ugmtEmuDqmDir+"/intermediate_muons"
# input histograms
errHistNumStr = 'errorSummaryNum'
errHistDenStr = 'errorSummaryDen'

# Muons
l1tStage2uGMTEmulatorCompRatioClient = cms.EDAnalyzer("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugmtEmuDEDqmDir),
    inputNum = cms.untracked.string(ugmtEmuDEDqmDir+'/'+errHistNumStr),
    inputDen = cms.untracked.string(ugmtEmuDEDqmDir+'/'+errHistDenStr),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT emulator muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)

# intermediate muons
titleStr = 'Summary of mismatch rates between uGMT intermediate muons and uGMT emulator intermediate muons from '
l1tStage2uGMTEmulImdMuBMTFCompRatioClient = l1tStage2uGMTEmulatorCompRatioClient.clone()
l1tStage2uGMTEmulImdMuBMTFCompRatioClient.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+'/BMTF/data_vs_emulator_comparison')
l1tStage2uGMTEmulImdMuBMTFCompRatioClient.inputNum = cms.untracked.string(ugmtEmuImdMuDqmDir+'/BMTF/data_vs_emulator_comparison/'+errHistNumStr)
l1tStage2uGMTEmulImdMuBMTFCompRatioClient.inputDen = cms.untracked.string(ugmtEmuImdMuDqmDir+'/BMTF/data_vs_emulator_comparison/'+errHistDenStr)
l1tStage2uGMTEmulImdMuBMTFCompRatioClient.ratioTitle = cms.untracked.string(titleStr+'BMTF')

l1tStage2uGMTEmulImdMuOMTFNegCompRatioClient = l1tStage2uGMTEmulatorCompRatioClient.clone()
l1tStage2uGMTEmulImdMuOMTFNegCompRatioClient.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_neg/data_vs_emulator_comparison')
l1tStage2uGMTEmulImdMuOMTFNegCompRatioClient.inputNum = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_neg/data_vs_emulator_comparison/'+errHistNumStr)
l1tStage2uGMTEmulImdMuOMTFNegCompRatioClient.inputDen = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_neg/data_vs_emulator_comparison/'+errHistDenStr)
l1tStage2uGMTEmulImdMuOMTFNegCompRatioClient.ratioTitle = cms.untracked.string(titleStr+'OMTF-')

l1tStage2uGMTEmulImdMuOMTFPosCompRatioClient = l1tStage2uGMTEmulatorCompRatioClient.clone()
l1tStage2uGMTEmulImdMuOMTFPosCompRatioClient.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_pos/data_vs_emulator_comparison')
l1tStage2uGMTEmulImdMuOMTFPosCompRatioClient.inputNum = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_pos/data_vs_emulator_comparison/'+errHistNumStr)
l1tStage2uGMTEmulImdMuOMTFPosCompRatioClient.inputDen = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_pos/data_vs_emulator_comparison/'+errHistDenStr)
l1tStage2uGMTEmulImdMuOMTFPosCompRatioClient.ratioTitle = cms.untracked.string(titleStr+'OMTF+')

l1tStage2uGMTEmulImdMuEMTFNegCompRatioClient = l1tStage2uGMTEmulatorCompRatioClient.clone()
l1tStage2uGMTEmulImdMuEMTFNegCompRatioClient.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+'/EMTF_neg/data_vs_emulator_comparison')
l1tStage2uGMTEmulImdMuEMTFNegCompRatioClient.inputNum = cms.untracked.string(ugmtEmuImdMuDqmDir+'/EMTF_neg/data_vs_emulator_comparison/'+errHistNumStr)
l1tStage2uGMTEmulImdMuEMTFNegCompRatioClient.inputDen = cms.untracked.string(ugmtEmuImdMuDqmDir+'/EMTF_neg/data_vs_emulator_comparison/'+errHistDenStr)
l1tStage2uGMTEmulImdMuEMTFNegCompRatioClient.ratioTitle = cms.untracked.string(titleStr+'EMTF-')

l1tStage2uGMTEmulImdMuEMTFPosCompRatioClient = l1tStage2uGMTEmulatorCompRatioClient.clone()
l1tStage2uGMTEmulImdMuEMTFPosCompRatioClient.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+'/EMTF_pos/data_vs_emulator_comparison')
l1tStage2uGMTEmulImdMuEMTFPosCompRatioClient.inputNum = cms.untracked.string(ugmtEmuImdMuDqmDir+'/EMTF_pos/data_vs_emulator_comparison/'+errHistNumStr)
l1tStage2uGMTEmulImdMuEMTFPosCompRatioClient.inputDen = cms.untracked.string(ugmtEmuImdMuDqmDir+'/EMTF_pos/data_vs_emulator_comparison/'+errHistDenStr)
l1tStage2uGMTEmulImdMuEMTFPosCompRatioClient.ratioTitle = cms.untracked.string(titleStr+'EMTF+')

# sequences
l1tStage2uGMTEmulatorClient = cms.Sequence(
    l1tStage2uGMTEmulatorCompRatioClient
  + l1tStage2uGMTEmulImdMuBMTFCompRatioClient
  + l1tStage2uGMTEmulImdMuOMTFNegCompRatioClient
  + l1tStage2uGMTEmulImdMuOMTFPosCompRatioClient
  + l1tStage2uGMTEmulImdMuEMTFNegCompRatioClient
  + l1tStage2uGMTEmulImdMuEMTFPosCompRatioClient
)

