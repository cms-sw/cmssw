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
caloStage2Params.egTrimmingLUTFile          = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egTrimmingLUT_10_v16.01.19.txt")
caloStage2Params.egMaxHcalEt                = cms.double(0.)
caloStage2Params.egMaxPtHOverE          = cms.double(128.)
caloStage2Params.egMaxHOverELUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/HoverEIdentification_0.995_v15.12.23.txt")
caloStage2Params.egCompressShapesLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCompressLUT_v3.txt")
caloStage2Params.egShapeIdType              = cms.string("compressed")
caloStage2Params.egShapeIdVersion           = cms.uint32(0)
caloStage2Params.egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/shapeIdentification_adapt0.99_compressedieta_compressedE_compressedshape_v15.12.08.txt")
caloStage2Params.egPUSType                  = cms.string("None")
caloStage2Params.egIsolationType            = cms.string("compressed")
caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/IsoIdentification_0.5_adapt_v16.02.01.txt")
caloStage2Params.egIsoAreaNrTowersEta       = cms.uint32(2)
caloStage2Params.egIsoAreaNrTowersPhi       = cms.uint32(4)
caloStage2Params.egIsoVetoNrTowersPhi       = cms.uint32(3)
#caloStage2Params.egIsoPUEstTowerGranularity = cms.uint32(1)
#caloStage2Params.egIsoMaxEtaAbsForTowerSum  = cms.uint32(4)
#caloStage2Params.egIsoMaxEtaAbsForIsoSum    = cms.uint32(27)
caloStage2Params.egPUSParams                = cms.vdouble(1,4,32) #Isolation window in firmware goes up to abs(ieta)=32 for now
caloStage2Params.egCalibrationType          = cms.string("compressed")
caloStage2Params.egCalibrationVersion       = cms.uint32(0)
caloStage2Params.egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/corrections_Trimming5_PU40bx25_compressedieta_compressedE_compressedshape_v15.11.27.txt")

# Tau
caloStage2Params.tauLsb                        = cms.double(0.5)
caloStage2Params.tauSeedThreshold              = cms.double(0.)
caloStage2Params.tauNeighbourThreshold         = cms.double(0.)
caloStage2Params.tauIsoAreaNrTowersEta         = cms.uint32(2)
caloStage2Params.tauIsoAreaNrTowersPhi         = cms.uint32(4)
caloStage2Params.tauIsoVetoNrTowersPhi         = cms.uint32(2)
caloStage2Params.tauPUSType                 = cms.string("None")
caloStage2Params.tauIsoLUTFile                 = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_iso_LUT_Option_4.txt")
caloStage2Params.tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/corrections_Trimming5_PU40bx25_woCALOEnergy_compressedieta_compressedE_L1Tau_hasEM_L1Tau_isMerged_v4.0.0.txt")
caloStage2Params.tauCompressLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Eta_Et_compression_LUT.txt")
caloStage2Params.tauPUSParams                  = cms.vdouble(1,4,27)

# jets
caloStage2Params.jetLsb                = cms.double(0.5)
caloStage2Params.jetSeedThreshold      = cms.double(1.5)
caloStage2Params.jetNeighbourThreshold = cms.double(0.)
caloStage2Params.jetPUSType            = cms.string("ChunkyDonut")

# Calibration options
# function6PtParams22EtaBins or None
#caloStage2Params.jetCalibrationType    = cms.string("function6PtParams22EtaBins")
#caloStage2Params.jetCalibrationType    = cms.string("None")
caloStage2Params.jetCalibrationType = cms.string("function6PtParams22EtaBins")


#Vector with 6 parameters for eta bin, from low eta to high
# 1,0,1,0,1,1 gives no correction
# must be in this form as may require > 255 arguments
jetCalibParamsVector = cms.vdouble()
jetCalibParamsVector.extend([
    1.44307778301,0,1,0,1,1, # Constant calibrations in HF bins
    1.26058321455,0,1,0,1,1,
    1.32478863889,0,1,0,1,1,
    1.45918510235,0,1,0,1,1,
    227.01736820,750.63078553,525.40739528,-227.14450318,0.00219807,2.52246411,
    227.03279291,759.30439917,520.09296374,-227.12919914,0.00355804,2.56065461,
    226.31030729,400.06847196,169.00860128,-227.20273887,0.00383241,2.57461063,
    19.07765721,424.12295555,10.39947925,-61.35187951,0.02158863,-1.54321595,
    9.95665741,19.07602422,2.17668279,-34.95753932,0.00218180,-20.65194585,
    9.17498863,18.56530947,2.36903187,-32.13403687,0.00211874,-20.97689108,
    4.90280506,30.43093246,2.99797836,-88.72826750,0.00491589,-20.31278701,
    4.90280506,30.43093246,2.99797836,-88.72826750,0.00491589,-20.31278701,
    9.17498863,18.56530947,2.36903187,-32.13403687,0.00211874,-20.97689108,
    9.95665741,19.07602422,2.17668279,-34.95753932,0.00218180,-20.65194585,
    19.07765721,424.12295555,10.39947925,-61.35187951,0.02158863,-1.54321595,
    226.31030729,400.06847196,169.00860128,-227.20273887,0.00383241,2.57461063,
    227.03279291,759.30439917,520.09296374,-227.12919914,0.00355804,2.56065461,
    227.01736820,750.63078553,525.40739528,-227.14450318,0.00219807,2.52246411,
    1.45918510235,0,1,0,1,1, # Constant calibrations in HF bins
    1.32478863889,0,1,0,1,1,
    1.26058321455,0,1,0,1,1,
    1.44307778301,0,1,0,1,1
])
caloStage2Params.jetCalibrationParams  = jetCalibParamsVector 

# sums: 0=ET, 1=HT, 2=MET, 3=MHT
caloStage2Params.etSumLsb                = cms.double(0.5)
caloStage2Params.etSumEtaMin             = cms.vint32(1, 1, 1, 1)
caloStage2Params.etSumEtaMax             = cms.vint32(28,  28,  28,  28)
caloStage2Params.etSumEtThreshold        = cms.vdouble(0.,  30.,  0.,  30.)

