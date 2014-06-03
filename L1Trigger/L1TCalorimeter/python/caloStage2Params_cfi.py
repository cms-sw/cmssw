import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloParams_cfi import caloParamsSource
import L1Trigger.L1TCalorimeter.caloParams_cfi
caloStage2Params = L1Trigger.L1TCalorimeter.caloParams_cfi.caloParams.clone()

# towers
caloStage2Params.towerLsbH        = cms.double(0.5)
caloStage2Params.towerLsbE        = cms.double(0.5)
caloStage2Params.towerLsbSum      = cms.double(0.5)
caloStage2Params.towerNBitsH      = cms.int32(8)
caloStage2Params.towerNBitsE      = cms.int32(8)
caloStage2Params.towerNBitsSum    = cms.int32(9)
caloStage2Params.towerNBitsRatio  = cms.int32(3)
caloStage2Params.towerEncoding    = cms.bool(False)

# regions
caloStage2Params.regionLsb        = cms.double(0.5)
caloStage2Params.regionPUSType    = cms.string("None")
caloStage2Params.regionPUSParams  = cms.vdouble()

# EG
caloStage2Params.egLsb                      = cms.double(0.5)
caloStage2Params.egSeedThreshold            = cms.double(2.)
caloStage2Params.egNeighbourThreshold       = cms.double(1.)
caloStage2Params.egHcalThreshold            = cms.double(1.)
caloStage2Params.egMaxHcalEt                = cms.double(0.)
caloStage2Params.egEtToRemoveHECut          = cms.double(128.)
caloStage2Params.egMaxHOverELUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egMaxHOverELUT.txt")
caloStage2Params.egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egShapeIdLUT.txt")
caloStage2Params.egIsoPUSType               = cms.string("None")
caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT_PU40bx25.txt")
caloStage2Params.egIsoAreaNrTowersEta       = cms.uint32(2)
caloStage2Params.egIsoAreaNrTowersPhi       = cms.uint32(4)
caloStage2Params.egIsoVetoNrTowersPhi       = cms.uint32(3)
caloStage2Params.egIsoPUEstTowerGranularity = cms.uint32(1)
caloStage2Params.egIsoMaxEtaAbsForTowerSum  = cms.uint32(4)
caloStage2Params.egIsoMaxEtaAbsForIsoSum    = cms.uint32(27)
caloStage2Params.egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCalibrationLUT.txt")

# Tau
caloStage2Params.tauLsb                = cms.double(0.5)
caloStage2Params.tauSeedThreshold      = cms.double(0.)
caloStage2Params.tauNeighbourThreshold = cms.double(0.)
caloStage2Params.tauIsoPUSType         = cms.string("None")
caloStage2Params.tauIsoLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauIsoLUT.txt")

# jets
caloStage2Params.jetLsb                = cms.double(0.5)
caloStage2Params.jetSeedThreshold      = cms.double(0.)
caloStage2Params.jetNeighbourThreshold = cms.double(0.)
caloStage2Params.jetPUSType            = cms.string("None")
caloStage2Params.jetCalibrationType    = cms.string("None")
caloStage2Params.jetCalibrationParams  = cms.vdouble()

# sums
caloStage2Params.etSumLsb                = cms.double(0.5)
caloStage2Params.etSumEtaMin             = cms.vint32(-999, -999, -999, -999)
caloStage2Params.etSumEtaMax             = cms.vint32(999,  999,  999,  999)
caloStage2Params.etSumEtThreshold        = cms.vdouble(0.,  0.,   0.,   0.)

