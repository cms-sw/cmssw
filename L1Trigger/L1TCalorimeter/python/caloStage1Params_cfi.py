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
caloStage1Params.egSeedThreshold      = cms.double(1.)

## not used anymore.  Isolation cut written in LUT filename
##caloStage1Params.egRelativeJetIsolationBarrelCut = cms.double(0.3)  # 0.3 for loose, 0.2 for tight
##caloStage1Params.egRelativeJetIsolationEndcapCut = cms.double(0.5)  # 0.5 for loose, 0.4 for tight

caloStage1Params.egMinPtRelativeJetIsolation = cms.int32(25)  
caloStage1Params.egMaxPtRelativeJetIsolation = cms.int32(63)  
caloStage1Params.egMinPt3x3HoE = cms.int32(1)  
caloStage1Params.egMaxPt3x3HoE = cms.int32(40)  

## EG Isolation LUT
caloStage1Params.egIsoLUTFileBarrel   = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT_stage1_isol0.30.txt")
caloStage1Params.egIsoLUTFileEndcaps  = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT_stage1_isol0.50.txt")

# Tau
caloStage1Params.tauSeedThreshold = cms.double(7.)
caloStage1Params.tauNeighbourThreshold = cms.double(0.)
#Tau parameters below are only used for setting tau isolation flag
caloStage1Params.switchOffTauVeto = cms.double(64.)
caloStage1Params.switchOffTauIso = cms.double(192.)
caloStage1Params.tauRelativeJetIsolationLimit  = cms.double(100.)
caloStage1Params.tauRelativeJetIsolationCut = cms.double(0.1)
caloStage1Params.tauIsoLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauIsoLUT.txt")

# jets
caloStage1Params.jetLsb                = cms.double(0.5)
caloStage1Params.jetSeedThreshold      = cms.double(10.)
caloStage1Params.jetNeighbourThreshold = cms.double(0.)
caloStage1Params.jetCalibrationType    = cms.string("Stage1JEC")
caloStage1Params.jetCalibrationParams  = jetSF_8TeV_data

# sums
caloStage1Params.etSumLsb                = cms.double(0.5)
caloStage1Params.etSumEtaMin             = cms.vint32(4, 4) #ET, HT
caloStage1Params.etSumEtaMax             = cms.vint32(17, 17) #ET, HT
caloStage1Params.etSumEtThreshold        = cms.vdouble(0., 7.) #ET, HT

