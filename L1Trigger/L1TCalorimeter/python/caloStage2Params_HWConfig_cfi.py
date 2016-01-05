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
caloStage2Params.towerEncoding    = cms.bool(True)

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
caloStage2Params.egCompressShapesLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCompressLUT_v1.txt")
caloStage2Params.egShapeIdType              = cms.string("compressed")
caloStage2Params.egShapeIdVersion           = cms.uint32(0)
caloStage2Params.egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/shapeIdentification_0.99_compressedieta_compressedE_compressedshape_v15.10.20.txt")
caloStage2Params.egPUSType                  = cms.string("None")
caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUTPU40bx25NrRings4Eff95.txt")
caloStage2Params.egIsoAreaNrTowersEta       = cms.uint32(2)
caloStage2Params.egIsoAreaNrTowersPhi       = cms.uint32(4)
caloStage2Params.egIsoVetoNrTowersPhi       = cms.uint32(3)
#caloStage2Params.egIsoPUEstTowerGranularity = cms.uint32(1)
#caloStage2Params.egIsoMaxEtaAbsForTowerSum  = cms.uint32(4)
#caloStage2Params.egIsoMaxEtaAbsForIsoSum    = cms.uint32(27)
caloStage2Params.egPUSParams                = cms.vdouble(1,4,27)
caloStage2Params.egCalibrationType          = cms.string("compressed")
caloStage2Params.egCalibrationVersion       = cms.uint32(0)
caloStage2Params.egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/corrections_Trimming5_PU40bx25_compressedieta_compressedE_compressedshape_v15.10.20.txt")

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
caloStage2Params.jetSeedThreshold      = cms.double(1.5)
caloStage2Params.jetNeighbourThreshold = cms.double(0.)
caloStage2Params.jetPUSType            = cms.string("ChunkyDonut")

#Calibration options 
# e.g. function6PtParams22EtaBins function6PtParams80EtaBins
#caloStage2Params.jetCalibrationType    = cms.string("function6PtParams80EtaBins")
caloStage2Params.jetCalibrationType    = cms.string("None")
#caloStage2Params.jetCalibrationType = cms.string("function6PtParams22EtaBins")


#Vector with 6 parameters for eta bin, from low eta to high
# 1,0,1,0,1,1 gives no correction
# must be in this form as may require > 255 arguments
jetCalibParamsVector = cms.vdouble() #Currently contains factors for function6PtParams80EtaBins 
jetCalibParamsVector.extend([
    1,0,1,0,1,1, # No calibrations in HF bins
    1,0,1,0,1,1,
    1,0,1,0,1,1,
    1,0,1,0,1,1,
    5.24246537,6.60700156,1.22785564,-13.69502129,0.00196905,-20.27233882,
    0.90833682,6.50791252,0.61922676,-209.49688550,0.01329731,-18.51593877,
    5.79849519,12.80862387,1.33405525,-25.10166231,0.00275828,-20.04923840,
    6.78385680,23.01868950,2.25627456,-39.95709157,0.00390259,-17.70111029,
    3.48234814,13.34746568,1.48348018,-46.10680359,0.00447602,-20.97512052,
    4.45650191,16.52912233,1.97499544,-41.54895663,0.00394956,-20.44045700,
    3.18556244,25.56760298,2.51677342,-103.26529010,0.00678420,-18.73657857,
    3.18556244,25.56760298,2.51677342,-103.26529010,0.00678420,-18.73657857,
    4.45650191,16.52912233,1.97499544,-41.54895663,0.00394956,-20.44045700,
    3.48234814,13.34746568,1.48348018,-46.10680359,0.00447602,-20.97512052,
    6.78385680,23.01868950,2.25627456,-39.95709157,0.00390259,-17.70111029,
    5.79849519,12.80862387,1.33405525,-25.10166231,0.00275828,-20.04923840,
    0.90833682,6.50791252,0.61922676,-209.49688550,0.01329731,-18.51593877,
    5.24246537,6.60700156,1.22785564,-13.69502129,0.00196905,-20.27233882,
    1,0,1,0,1,1, # No calibrations in HF bins
    1,0,1,0,1,1,
    1,0,1,0,1,1,
    1,0,1,0,1,1
])
caloStage2Params.jetCalibrationParams  = jetCalibParamsVector 

# sums: 0=ET, 1=HT, 2=MET, 3=MHT
caloStage2Params.etSumLsb                = cms.double(0.5)
caloStage2Params.etSumEtaMin             = cms.vint32(-40, -36, -40, -36)
caloStage2Params.etSumEtaMax             = cms.vint32(40,  36,  40,  36)
caloStage2Params.etSumEtThreshold        = cms.vdouble(0.,  0.,   0.,   0.)

