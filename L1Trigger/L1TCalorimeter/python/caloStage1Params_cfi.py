import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloStage1RegionSF_cfi import *
from L1Trigger.L1TCalorimeter.caloStage1JetSF_cfi import *

from L1Trigger.L1TCalorimeter.caloParams_cfi import caloParamsSource
import L1Trigger.L1TCalorimeter.caloParams_cfi
#caloStage1ParamsSource = L1Trigger.L1TCalorimeter.caloParams_cfi.caloParamsSource.clone()
caloStage1Params = L1Trigger.L1TCalorimeter.caloParams_cfi.caloParams.clone()

caloStage1Params.regionPUSType    = cms.string("PUM0")       #"None" for no PU subtraction, "PUM0", "HICaloRingSub"
caloStage1Params.regionPUSParams  = regionSubtraction_PU40_MC13TeV

# EG
caloStage1Params.egLsb                = cms.double(1.)
caloStage1Params.egSeedThreshold      = cms.double(0.)

caloStage1Params.egMinPtJetIsolation = cms.int32(25)
caloStage1Params.egMaxPtJetIsolation = cms.int32(63)
caloStage1Params.egMinPtHOverEIsolation = cms.int32(1)
caloStage1Params.egMaxPtHOverEIsolation = cms.int32(40)

caloStage1Params.egPUSType    = cms.string("None")
caloStage1Params.egPUSParams  = cms.vdouble()

## EG Isolation LUT
## caloStage1Params.egIsoLUTFile   = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT_stage1.txt")
caloStage1Params.egIsoLUTFile      = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT_stage1_isolEB0.30_isolEE0.50_combined.txt")
#caloStage1Params.egIsoLUTFileBarrel   = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT_stage1_isol0.30.txt")
#caloStage1Params.egIsoLUTFileEndcaps  = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT_stage1_isol0.50.txt")

# Tau
caloStage1Params.tauSeedThreshold = cms.double(7.)
caloStage1Params.tauNeighbourThreshold = cms.double(0.)
#Tau parameters below are only used for setting tau isolation flag
caloStage1Params.tauMaxPtTauVeto = cms.double(64.)
caloStage1Params.tauMinPtJetIsolationB = cms.double(192.)
caloStage1Params.tauMaxJetIsolationB  = cms.double(100.)
caloStage1Params.tauMaxJetIsolationA = cms.double(0.1)
caloStage1Params.tauIsoLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauIsoLUT_stage1_isolA0.10_isolB100.00_ch_switchToIsoBPt192.00_j8t8.txt")
## caloStage1Params.tauCalibrationLUTFile = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUT_stage1.txt")
caloStage1Params.tauCalibrationLUTFile = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauL1Calib_LUT.txt")
caloStage1Params.tauEtToHFRingEtLUTFile= cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauHwEtToHFRingScale_LUT.txt")
caloStage1Params.isoTauEtaMin          = cms.int32(5)
caloStage1Params.isoTauEtaMax          = cms.int32(16)
# jets
caloStage1Params.jetLsb                = cms.double(0.5)
caloStage1Params.jetSeedThreshold      = cms.double(5.)
caloStage1Params.jetNeighbourThreshold = cms.double(0.)
caloStage1Params.jetCalibrationType    = cms.string("Stage1JEC")
caloStage1Params.jetCalibrationParams  = jetSF_8TeV_data
## caloStage1Params.jetCalibrationLUTFile = cms.FileInPath("L1Trigger/L1TCalorimeter/data/jetCalibrationLUT_stage1_prelim.txt")
caloStage1Params.jetCalibrationLUTFile = cms.FileInPath("L1Trigger/L1TCalorimeter/data/jetCalibrationLUT_symmetric_0is0.txt")

# sums
caloStage1Params.etSumLsb                = cms.double(0.5)
caloStage1Params.etSumEtaMin             = cms.vint32(4, 4) #ET, HT
caloStage1Params.etSumEtaMax             = cms.vint32(17, 17) #ET, HT
caloStage1Params.etSumEtThreshold        = cms.vdouble(0., 7.) #ET, HT

# HI
caloStage1Params.centralityLUTFile = cms.FileInPath("L1Trigger/L1TCalorimeter/data/centralityLUT_stage1.txt")
caloStage1Params.q2LUTFile = cms.FileInPath("L1Trigger/L1TCalorimeter/data/q2LUT_stage1.txt")
