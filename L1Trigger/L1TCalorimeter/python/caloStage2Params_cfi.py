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
caloStage2Params.egHcalThreshold            = cms.double(0.)
caloStage2Params.egTrimmingLUTFile          = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egTrimmingLUT_5.txt")
caloStage2Params.egMaxHcalEt                = cms.double(0.)
caloStage2Params.egEtToRemoveHECut          = cms.double(128.)
caloStage2Params.egMaxHOverELUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egMaxHOverELUT_995eff.txt")
caloStage2Params.egCompressShapesLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCompressShapesLUT.txt")
caloStage2Params.egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egShapeIdLUT_995eff.txt")
caloStage2Params.egPUSType               = cms.string("None")
caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUTPU40bx25NrRings4Eff95.txt")
caloStage2Params.egIsoAreaNrTowersEta       = cms.uint32(2)
caloStage2Params.egIsoAreaNrTowersPhi       = cms.uint32(4)
caloStage2Params.egIsoVetoNrTowersPhi       = cms.uint32(3)
#caloStage2Params.egIsoPUEstTowerGranularity = cms.uint32(1)
#caloStage2Params.egIsoMaxEtaAbsForTowerSum  = cms.uint32(4)
#caloStage2Params.egIsoMaxEtaAbsForIsoSum    = cms.uint32(27)
caloStage2Params.egPUSParams                = cms.vdouble(1,4,27)

caloStage2Params.egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCalibrationLUT_Trimming5.txt")

# Tau
caloStage2Params.tauLsb                        = cms.double(0.5)
caloStage2Params.tauSeedThreshold              = cms.double(0.)
caloStage2Params.tauNeighbourThreshold         = cms.double(0.)
caloStage2Params.tauIsoPUSType                 = cms.string("None")
caloStage2Params.tauIsoLUTFile                 = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauIsoLUT.txt")
caloStage2Params.tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUT.txt")

# jets
caloStage2Params.jetLsb                = cms.double(0.5)
caloStage2Params.jetSeedThreshold      = cms.double(0.)
caloStage2Params.jetNeighbourThreshold = cms.double(0.)
caloStage2Params.jetPUSType            = cms.string("None")
#Calibration options 
# e.g. function6PtParams22EtaBins function6PtParams80EtaBins
caloStage2Params.jetCalibrationType    = cms.string("function6PtParams80EtaBins")
#Vector with 6 parameters for eta bin, from low eta to high
# 1,0,1,0,1,1 gives no correction
# must be in this form as may require > 255 arguments
jetCalibParamsVector = cms.vdouble()
jetCalibParamsVector.extend([
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta1 eta2
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta3 eta4
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta5 eta6
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta7 eta8
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta9 eta2
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta11 eta12
    1.117222,	0.115154,	1.910615,	1.013722,	0.141106,	0.932544,
    1.117222,	0.115154,	1.910615,	1.013722,	0.141106,	0.932544,
    1.117222,	0.115154,	1.910615,	1.013722,	0.141106,	0.932544,
    1.117222,	0.115154,	1.910615,	1.013722,	0.141106,	0.932544,
    1.117222,	0.115154,	1.910615,	1.013722,	0.141106,	0.932544,
    1.117222,	0.115154,	1.910615,	1.013722,	0.141106,	0.932544,
    1.117222,	0.115154,	1.910615,	1.013722,	0.141106,	0.932544,
      1.097168,	3.065602,	-1.000000,	0.769128,	0.237118,	2.149065,
      1.097168,	3.065602,	-1.000000,	0.769128,	0.237118,	2.149065,
      1.097168,	3.065602,	-1.000000,	0.769128,	0.237118,	2.149065,
      1.097168,	3.065602,	-1.000000,	0.769128,	0.237118,	2.149065,
      1.097168,	3.065602,	-1.000000,	0.769128,	0.237118,	2.149065,
      1.097168,	3.065602,	-1.000000,	0.769128,	0.237118,	2.149065,
      1.097168,	3.065602,	-1.000000,	0.769128,	0.237118,	2.149065,
      1.136084,	4.306006,	-1.000000,	0.560253,	0.387929,	2.712581,
      1.136084,	4.306006,	-1.000000,	0.560253,	0.387929,	2.712581,
      1.136084,	4.306006,	-1.000000,	0.560253,	0.387929,	2.712581,
      1.136084,	4.306006,	-1.000000,	0.560253,	0.387929,	2.712581,
      1.136084,	4.306006,	-1.000000,	0.560253,	0.387929,	2.712581,
      1.136084,	4.306006,	-1.000000,	0.560253,	0.387929,	2.712581,
      1.136084,	4.306006,	-1.000000,	0.560253,	0.387929,	2.712581,
      1.120055,	1.720310,	-1.000000,	0.701081,	0.236731,	2.260849,
      1.120055,	1.720310,	-1.000000,	0.701081,	0.236731,	2.260849,
      1.120055,	1.720310,	-1.000000,	0.701081,	0.236731,	2.260849,
      1.120055,	1.720310,	-1.000000,	0.701081,	0.236731,	2.260849,
      1.120055,	1.720310,	-1.000000,	0.701081,	0.236731,	2.260849,
      1.120055,	1.720310,	-1.000000,	0.701081,	0.236731,	2.260849,
      1.120055,	1.720310,	-1.000000,	0.701081,	0.236731,	2.260849,

                                        ])

jetCalibParamsVector.extend([
      1.147088,	0.000000,	2.021120,	1.163963,	0.157996,	1.465768,
      1.147088,	0.000000,	2.021120,	1.163963,	0.157996,	1.465768,
      1.147088,	0.000000,	2.021120,	1.163963,	0.157996,	1.465768,
      1.147088,	0.000000,	2.021120,	1.163963,	0.157996,	1.465768,
      1.147088,	0.000000,	2.021120,	1.163963,	0.157996,	1.465768,
      1.147088,	0.000000,	2.021120,	1.163963,	0.157996,	1.465768,
      1.147088,	0.000000,	2.021120,	1.163963,	0.157996,	1.465768,
      1.126069,	4.677360,	-1.000000,	0.507453,	0.411144,	2.794454,
      1.126069,	4.677360,	-1.000000,	0.507453,	0.411144,	2.794454,
      1.126069,	4.677360,	-1.000000,	0.507453,	0.411144,	2.794454,
      1.126069,	4.677360,	-1.000000,	0.507453,	0.411144,	2.794454,
      1.126069,	4.677360,	-1.000000,	0.507453,	0.411144,	2.794454,
      1.126069,	4.677360,	-1.000000,	0.507453,	0.411144,	2.794454,
      1.126069,	4.677360,	-1.000000,	0.507453,	0.411144,	2.794454,
      1.089151,	3.979116,	-1.000000,	0.595467,	0.288850,	2.478056,
      1.089151,	3.979116,	-1.000000,	0.595467,	0.288850,	2.478056,
      1.089151,	3.979116,	-1.000000,	0.595467,	0.288850,	2.478056,
      1.089151,	3.979116,	-1.000000,	0.595467,	0.288850,	2.478056,
      1.089151,	3.979116,	-1.000000,	0.595467,	0.288850,	2.478056,
      1.089151,	3.979116,	-1.000000,	0.595467,	0.288850,	2.478056,
      1.089151,	3.979116,	-1.000000,	0.595467,	0.288850,	2.478056,
      1.135149,	0.010747,	7.372626,	0.961446,	0.153867,	1.104507,
      1.135149,	0.010747,	7.372626,	0.961446,	0.153867,	1.104507,
      1.135149,	0.010747,	7.372626,	0.961446,	0.153867,	1.104507,
      1.135149,	0.010747,	7.372626,	0.961446,	0.153867,	1.104507,
      1.135149,	0.010747,	7.372626,	0.961446,	0.153867,	1.104507,
      1.135149,	0.010747,	7.372626,	0.961446,	0.153867,	1.104507,
      1.135149,	0.010747,	7.372626,	0.961446,	0.153867,	1.104507,
                                        1,0,1,0,1,1, 1,0,1,0,1,1, 
                                        1,0,1,0,1,1, 1,0,1,0,1,1, 
                                        1,0,1,0,1,1, 1,0,1,0,1,1, 
                                        1,0,1,0,1,1, 1,0,1,0,1,1, 
                                        1,0,1,0,1,1, 1,0,1,0,1,1, 
                                        1,0,1,0,1,1, 1,0,1,0,1,1, 
                                       ] )
caloStage2Params.jetCalibrationParams  = jetCalibParamsVector 

# sums
caloStage2Params.etSumLsb                = cms.double(0.5)
caloStage2Params.etSumEtaMin             = cms.vint32(-999, -999, -999, -999)
caloStage2Params.etSumEtaMax             = cms.vint32(999,  999,  999,  999)
caloStage2Params.etSumEtThreshold        = cms.vdouble(0.,  0.,   0.,   0.)

