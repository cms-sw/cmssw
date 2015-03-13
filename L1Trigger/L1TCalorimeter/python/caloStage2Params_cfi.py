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
caloStage2Params.egMaxPtHOverE          = cms.double(128.)
caloStage2Params.egMaxHOverELUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egMaxHOverELUT_995eff.txt")
caloStage2Params.egCompressShapesLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCompressShapesLUT.txt")
caloStage2Params.egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egShapeIdLUT_995eff.txt")
caloStage2Params.egPUSType                  = cms.string("None")
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
caloStage2Params.tauIsoAreaNrTowersEta         = cms.uint32(2)
caloStage2Params.tauIsoAreaNrTowersPhi         = cms.uint32(4)
caloStage2Params.tauIsoVetoNrTowersPhi         = cms.uint32(2)
caloStage2Params.tauPUSType                 = cms.string("None")
caloStage2Params.tauIsoLUTFile                 = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauIsoLUTetPU.txt")
caloStage2Params.tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUT.txt")
caloStage2Params.tauPUSParams                  = cms.vdouble(1,4,27)

# jets
caloStage2Params.jetLsb                = cms.double(0.5)
caloStage2Params.jetSeedThreshold      = cms.double(0.)
caloStage2Params.jetNeighbourThreshold = cms.double(0.)
caloStage2Params.jetPUSType            = cms.string("ChunkyDonut")

#Calibration options 
# e.g. function6PtParams22EtaBins function6PtParams80EtaBins
#caloStage2Params.jetCalibrationType    = cms.string("function6PtParams80EtaBins")
caloStage2Params.jetCalibrationType    = cms.string("None")

#Vector with 6 parameters for eta bin, from low eta to high
# 1,0,1,0,1,1 gives no correction
# must be in this form as may require > 255 arguments
jetCalibParamsVector = cms.vdouble() #Currently contains factors for function6PtParams80EtaBins 
jetCalibParamsVector.extend([
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta1 eta2
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta3 eta4
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta5 eta6
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta7 eta8
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta9 eta2
                                        1,0,1,0,1,1, 1,0,1,0,1,1, #eta11 eta12
      1.162001,	0.000000,	0.506532,	0.660991,	0.313898,	2.431404,
      1.162001,	0.000000,	0.506532,	0.660991,	0.313898,	2.431404,
      1.162001,	0.000000,	0.506532,	0.660991,	0.313898,	2.431404,
      1.162001,	0.000000,	0.506532,	0.660991,	0.313898,	2.431404,
      1.162001,	0.000000,	0.506532,	0.660991,	0.313898,	2.431404,
      1.162001,	0.000000,	0.506532,	0.660991,	0.313898,	2.431404,
      1.162001,	0.000000,	0.506532,	0.660991,	0.313898,	2.431404,
      1.195166,	0.000000,	3.462064,	1.500481,	0.217480,	1.860866,
      1.195166,	0.000000,	3.462064,	1.500481,	0.217480,	1.860866,
      1.195166,	0.000000,	3.462064,	1.500481,	0.217480,	1.860866,
      1.195166,	0.000000,	3.462064,	1.500481,	0.217480,	1.860866,
      1.195166,	0.000000,	3.462064,	1.500481,	0.217480,	1.860866,
      1.195166,	0.000000,	3.462064,	1.500481,	0.217480,	1.860866,
      1.195166,	0.000000,	3.462064,	1.500481,	0.217480,	1.860866,
      1.268926,	0.000000,	9.016650,	1.562618,	0.264561,	2.044610,
      1.268926,	0.000000,	9.016650,	1.562618,	0.264561,	2.044610,
      1.268926,	0.000000,	9.016650,	1.562618,	0.264561,	2.044610,
      1.268926,	0.000000,	9.016650,	1.562618,	0.264561,	2.044610,
      1.268926,	0.000000,	9.016650,	1.562618,	0.264561,	2.044610,
      1.268926,	0.000000,	9.016650,	1.562618,	0.264561,	2.044610,
      1.268926,	0.000000,	9.016650,	1.562618,	0.264561,	2.044610,
      1.162806,	0.703037,	-0.998253,	1.440402,	0.203554,	1.713803,
      1.162806,	0.703037,	-0.998253,	1.440402,	0.203554,	1.713803,
      1.162806,	0.703037,	-0.998253,	1.440402,	0.203554,	1.713803,
      1.162806,	0.703037,	-0.998253,	1.440402,	0.203554,	1.713803,
      1.162806,	0.703037,	-0.998253,	1.440402,	0.203554,	1.713803,
      1.162806,	0.703037,	-0.998253,	1.440402,	0.203554,	1.713803,
      1.162806,	0.703037,	-0.998253,	1.440402,	0.203554,	1.713803,


                                        ])

jetCalibParamsVector.extend([
      1.168191,	0.339318,	-0.925540,	1.539332,	0.191717,	1.600152,
      1.168191,	0.339318,	-0.925540,	1.539332,	0.191717,	1.600152,
      1.168191,	0.339318,	-0.925540,	1.539332,	0.191717,	1.600152,
      1.168191,	0.339318,	-0.925540,	1.539332,	0.191717,	1.600152,
      1.168191,	0.339318,	-0.925540,	1.539332,	0.191717,	1.600152,
      1.168191,	0.339318,	-0.925540,	1.539332,	0.191717,	1.600152,
      1.168191,	0.339318,	-0.925540,	1.539332,	0.191717,	1.600152,
      1.267062,	0.000000,	-0.951295,	1.551194,	0.259202,	2.032236,
      1.267062,	0.000000,	-0.951295,	1.551194,	0.259202,	2.032236,
      1.267062,	0.000000,	-0.951295,	1.551194,	0.259202,	2.032236,
      1.267062,	0.000000,	-0.951295,	1.551194,	0.259202,	2.032236,
      1.267062,	0.000000,	-0.951295,	1.551194,	0.259202,	2.032236,
      1.267062,	0.000000,	-0.951295,	1.551194,	0.259202,	2.032236,
      1.267062,	0.000000,	-0.951295,	1.551194,	0.259202,	2.032236,
      1.205309,	0.000248,	7.230826,	1.517030,	0.214394,	1.842128,
      1.205309,	0.000248,	7.230826,	1.517030,	0.214394,	1.842128,
      1.205309,	0.000248,	7.230826,	1.517030,	0.214394,	1.842128,
      1.205309,	0.000248,	7.230826,	1.517030,	0.214394,	1.842128,
      1.205309,	0.000248,	7.230826,	1.517030,	0.214394,	1.842128,
      1.205309,	0.000248,	7.230826,	1.517030,	0.214394,	1.842128,
      1.205309,	0.000248,	7.230826,	1.517030,	0.214394,	1.842128,
      1.174576,	0.156892,	9.999552,	0.635096,	0.343184,	2.509776,
      1.174576,	0.156892,	9.999552,	0.635096,	0.343184,	2.509776,
      1.174576,	0.156892,	9.999552,	0.635096,	0.343184,	2.509776,
      1.174576,	0.156892,	9.999552,	0.635096,	0.343184,	2.509776,
      1.174576,	0.156892,	9.999552,	0.635096,	0.343184,	2.509776,
      1.174576,	0.156892,	9.999552,	0.635096,	0.343184,	2.509776,
      1.174576,	0.156892,	9.999552,	0.635096,	0.343184,	2.509776,
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
caloStage2Params.etSumEtaMin             = cms.vint32(-999, -28, -999, -28)
caloStage2Params.etSumEtaMax             = cms.vint32(999,  28,  999,  28)
caloStage2Params.etSumEtThreshold        = cms.vdouble(0.,  30.,   0.,   30.)

