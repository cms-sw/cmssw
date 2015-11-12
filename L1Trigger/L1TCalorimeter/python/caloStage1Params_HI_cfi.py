import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloParams_cfi import caloParamsSource
import L1Trigger.L1TCalorimeter.caloParams_cfi
caloStage1Params = L1Trigger.L1TCalorimeter.caloParams_cfi.caloParams.clone()

caloStage1Params.ppRecord = cms.bool(False)
### nominal settings 2015-11-10
### PUS mask
caloStage1Params.jetRegionMask = cms.int32(0b0000100000000000010000)
### EG 'iso' (eta) mask
caloStage1Params.egEtaCut = cms.int32(0b0000001111111111000000)
### Single track eta mask
caloStage1Params.tauRegionMask = cms.int32(0b1111111100000011111111)
### Centrality eta mask
caloStage1Params.centralityRegionMask = cms.int32(0b0000111111111111110000)
### jet seed threshold for 3x3 step of jet finding
caloStage1Params.jetSeedThreshold = cms.double(0)
### HTT settings
caloStage1Params.etSumEtThreshold = cms.vdouble(0., 7.) #ET, HT
### Minimum Bias thresholds
caloStage1Params.minimumBiasThresholds = cms.vint32(4,4,6,6)
### Centrality LUT
caloStage1Params.centralityLUTFile = cms.FileInPath("L1Trigger/L1TCalorimeter/data/centrality_extended_LUT_preRun.txt")

#### not used, but necessary to set something to stop ESProducer error
caloStage1Params.egPUSParams = cms.vdouble()
caloStage1Params.etSumEtaMin = cms.vint32(4, 4) #ET, HT
caloStage1Params.etSumEtaMax = cms.vint32(17, 17) #ET, HT
