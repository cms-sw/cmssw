import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.L1TMonitor.L1TdeStage2uGMT_cff import ignoreFinalsBinsRun3, ignoreIntermediatesBins, ignoreIntermediatesBinsRun3

# directories
ugmtEmuDqmDir = "L1TEMU/L1TdeStage2uGMT"
ugmtEmuDEDqmDir = ugmtEmuDqmDir+"/data_vs_emulator_comparison"
ugmtEmuImdMuDqmDir = ugmtEmuDqmDir+"/intermediate_muons"
# input histograms
errHistNumStr = 'errorSummaryNum'
errHistDenStr = 'errorSummaryDen'

# Muons
l1tStage2uGMTEmulatorCompRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugmtEmuDEDqmDir),
    inputNum = cms.untracked.string(ugmtEmuDEDqmDir+'/'+errHistNumStr),
    inputDen = cms.untracked.string(ugmtEmuDEDqmDir+'/'+errHistDenStr),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT muons and uGMT emulator muons'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True)
)
## Era: Run3_2021; Ignore BX range mismatches. This is necessary because we only read out the central BX for the inputs, so that is what the emulator has to work on.
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(l1tStage2uGMTEmulatorCompRatioClient, ignoreBin = cms.untracked.vint32(ignoreFinalsBinsRun3))

# Showers
l1tStage2uGMTShowerEmulatorCompRatioClient = DQMEDHarvester("L1TStage2RatioClient",
    monitorDir = cms.untracked.string(ugmtEmuDEDqmDir+"/Muon showers"),
    inputNum = cms.untracked.string(ugmtEmuDEDqmDir+'/Muon showers/'+errHistNumStr),
    inputDen = cms.untracked.string(ugmtEmuDEDqmDir+'/Muon showers/'+errHistDenStr),
    ratioName = cms.untracked.string('mismatchRatio'),
    ratioTitle = cms.untracked.string('Summary of mismatch rates between uGMT showers and uGMT emulator showers'),
    yAxisTitle = cms.untracked.string('# mismatch / # total'),
    binomialErr = cms.untracked.bool(True),
    ignoreBin = cms.untracked.vint32(ignoreFinalsBinsRun3), # Ignore BX range mismatches. This is necessary because we only read out the central BX for the inputs, so that is what the emulator has to work on.
)

# intermediate muons
titleStr = 'Summary of mismatch rates between uGMT intermediate muons and uGMT emulator intermediate muons from '
l1tStage2uGMTEmulImdMuBMTFCompRatioClient = l1tStage2uGMTEmulatorCompRatioClient.clone()
l1tStage2uGMTEmulImdMuBMTFCompRatioClient.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+'/BMTF/data_vs_emulator_comparison')
l1tStage2uGMTEmulImdMuBMTFCompRatioClient.inputNum = cms.untracked.string(ugmtEmuImdMuDqmDir+'/BMTF/data_vs_emulator_comparison/'+errHistNumStr)
l1tStage2uGMTEmulImdMuBMTFCompRatioClient.inputDen = cms.untracked.string(ugmtEmuImdMuDqmDir+'/BMTF/data_vs_emulator_comparison/'+errHistDenStr)
l1tStage2uGMTEmulImdMuBMTFCompRatioClient.ratioTitle = cms.untracked.string(titleStr+'BMTF')
l1tStage2uGMTEmulImdMuBMTFCompRatioClient.ignoreBin = cms.untracked.vint32(ignoreIntermediatesBins)
## Era: Run3_2021; Ignore BX range mismatches. This is necessary because we only read out the central BX for the inputs, so that is what the emulator has to work on.
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(l1tStage2uGMTEmulImdMuBMTFCompRatioClient, ignoreBin = ignoreIntermediatesBinsRun3)

l1tStage2uGMTEmulImdMuOMTFNegCompRatioClient = l1tStage2uGMTEmulImdMuBMTFCompRatioClient.clone()
l1tStage2uGMTEmulImdMuOMTFNegCompRatioClient.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_neg/data_vs_emulator_comparison')
l1tStage2uGMTEmulImdMuOMTFNegCompRatioClient.inputNum = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_neg/data_vs_emulator_comparison/'+errHistNumStr)
l1tStage2uGMTEmulImdMuOMTFNegCompRatioClient.inputDen = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_neg/data_vs_emulator_comparison/'+errHistDenStr)
l1tStage2uGMTEmulImdMuOMTFNegCompRatioClient.ratioTitle = cms.untracked.string(titleStr+'OMTF-')

l1tStage2uGMTEmulImdMuOMTFPosCompRatioClient = l1tStage2uGMTEmulImdMuBMTFCompRatioClient.clone()
l1tStage2uGMTEmulImdMuOMTFPosCompRatioClient.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_pos/data_vs_emulator_comparison')
l1tStage2uGMTEmulImdMuOMTFPosCompRatioClient.inputNum = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_pos/data_vs_emulator_comparison/'+errHistNumStr)
l1tStage2uGMTEmulImdMuOMTFPosCompRatioClient.inputDen = cms.untracked.string(ugmtEmuImdMuDqmDir+'/OMTF_pos/data_vs_emulator_comparison/'+errHistDenStr)
l1tStage2uGMTEmulImdMuOMTFPosCompRatioClient.ratioTitle = cms.untracked.string(titleStr+'OMTF+')

l1tStage2uGMTEmulImdMuEMTFNegCompRatioClient = l1tStage2uGMTEmulImdMuBMTFCompRatioClient.clone()
l1tStage2uGMTEmulImdMuEMTFNegCompRatioClient.monitorDir = cms.untracked.string(ugmtEmuImdMuDqmDir+'/EMTF_neg/data_vs_emulator_comparison')
l1tStage2uGMTEmulImdMuEMTFNegCompRatioClient.inputNum = cms.untracked.string(ugmtEmuImdMuDqmDir+'/EMTF_neg/data_vs_emulator_comparison/'+errHistNumStr)
l1tStage2uGMTEmulImdMuEMTFNegCompRatioClient.inputDen = cms.untracked.string(ugmtEmuImdMuDqmDir+'/EMTF_neg/data_vs_emulator_comparison/'+errHistDenStr)
l1tStage2uGMTEmulImdMuEMTFNegCompRatioClient.ratioTitle = cms.untracked.string(titleStr+'EMTF-')

l1tStage2uGMTEmulImdMuEMTFPosCompRatioClient = l1tStage2uGMTEmulImdMuBMTFCompRatioClient.clone()
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

# Add shower tests for Run3
_run3_l1tStage2uGMTEmulatorClient = cms.Sequence(l1tStage2uGMTEmulatorClient.copy() + l1tStage2uGMTShowerEmulatorCompRatioClient)
stage2L1Trigger_2021.toReplaceWith(l1tStage2uGMTEmulatorClient, _run3_l1tStage2uGMTEmulatorClient)
