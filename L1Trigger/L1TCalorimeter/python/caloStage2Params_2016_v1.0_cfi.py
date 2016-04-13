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
caloStage2Params.egCompressShapesLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCompressLUT_v4.txt")
caloStage2Params.egShapeIdType              = cms.string("compressed")
caloStage2Params.egShapeIdVersion           = cms.uint32(0)
caloStage2Params.egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/shapeIdentification_adapt0.99_compressedieta_compressedE_compressedshape_v15.12.08.txt")
caloStage2Params.egPUSType                  = cms.string("None")
caloStage2Params.egIsolationType            = cms.string("compressed")
caloStage2Params.egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/IsoIdentification_0.25_adapt_extrap_EBEE_v16.04.04.txt")
caloStage2Params.egIsoAreaNrTowersEta       = cms.uint32(2)
caloStage2Params.egIsoAreaNrTowersPhi       = cms.uint32(4)
caloStage2Params.egIsoVetoNrTowersPhi       = cms.uint32(3)
#caloStage2Params.egIsoPUEstTowerGranularity = cms.uint32(1)
#caloStage2Params.egIsoMaxEtaAbsForTowerSum  = cms.uint32(4)
#caloStage2Params.egIsoMaxEtaAbsForIsoSum    = cms.uint32(27)
caloStage2Params.egPUSParams                = cms.vdouble(1,4,32) #Isolation window in firmware goes up to abs(ieta)=32 for now
caloStage2Params.egCalibrationType          = cms.string("compressed")
caloStage2Params.egCalibrationVersion       = cms.uint32(0)
caloStage2Params.egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/corrections_Trimming5_compressedieta_compressedE_compressedshape_v16.04.01.txt")

# Tau
caloStage2Params.tauLsb                        = cms.double(0.5)
caloStage2Params.tauSeedThreshold              = cms.double(0.)
caloStage2Params.tauNeighbourThreshold         = cms.double(0.)
caloStage2Params.tauIsoAreaNrTowersEta         = cms.uint32(2)
caloStage2Params.tauIsoAreaNrTowersPhi         = cms.uint32(4)
caloStage2Params.tauIsoVetoNrTowersPhi         = cms.uint32(2)
caloStage2Params.tauPUSType                    = cms.string("None")
caloStage2Params.tauIsoLUTFile                 = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_21_NoLayer1Calibration_noCompressionBlock_v4.0.0.txt")
caloStage2Params.tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Calibration_LUT_NoLayer1Calibration_v9.0.0.txt")
caloStage2Params.tauCompressLUTFile            = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCompressAllLUT_12bit_v3.txt")
caloStage2Params.tauPUSParams                  = cms.vdouble(1,4,32)

# jets
caloStage2Params.jetLsb                = cms.double(0.5)
caloStage2Params.jetSeedThreshold      = cms.double(1.5)
caloStage2Params.jetNeighbourThreshold = cms.double(0.)
caloStage2Params.jetPUSType            = cms.string("ChunkyDonut")

# Calibration options
# function6PtParams22EtaBins or None
#caloStage2Params.jetCalibrationType    = cms.string("None")
caloStage2Params.jetCalibrationType = cms.string("function6PtParams22EtaBins")


#Vector with 6 parameters for eta bin, from low eta to high
# 1,0,1,0,1,1 gives no correction
# must be in this form as may require > 255 arguments
jetCalibParamsVector = cms.vdouble()
jetCalibParamsVector.extend([
    1.52819258712,0,1,0,1,1, # Constant calibrations in HF bins
    1.32880934866,0,1,0,1,1,
    1.35650073562,0,1,0,1,1,
    1.55766262213,0,1,0,1,1,
    12.42641269,6.35085569,1.21845374,-14.70772743,0.00118945,-10.56931753,
    12.14768583,8.05320174,0.92028575,-15.48969055,0.00155426,-10.48022143,
    11.87301715,10.45078171,1.02636603,-16.87586292,0.00210878,-10.34892982,
    12.09471152,9.36873133,0.96484984,-16.09937338,0.00177759,-10.43545594,
    4.69610083,11.50217825,1.31071693,-23.88756705,0.00336036,-19.28130829,
    7.08667633,23.68511274,2.52417732,-43.09154102,0.00359229,-18.80055010,
    3.44975454,28.42124526,2.58428891,-137.42027137,0.00669104,-19.53448313,
    3.44975454,28.42124526,2.58428891,-137.42027137,0.00669104,-19.53448313,
    7.08667633,23.68511274,2.52417732,-43.09154102,0.00359229,-18.80055010,
    4.69610083,11.50217825,1.31071693,-23.88756705,0.00336036,-19.28130829,
    12.09471152,9.36873133,0.96484984,-16.09937338,0.00177759,-10.43545594,
    11.87301715,10.45078171,1.02636603,-16.87586292,0.00210878,-10.34892982,
    12.14768583,8.05320174,0.92028575,-15.48969055,0.00155426,-10.48022143,
    12.42641269,6.35085569,1.21845374,-14.70772743,0.00118945,-10.56931753,
    1.55766262213,0,1,0,1,1, # Constant calibrations in HF bins
    1.35650073562,0,1,0,1,1,
    1.32880934866,0,1,0,1,1,
    1.52819258712,0,1,0,1,1
])
caloStage2Params.jetCalibrationParams  = jetCalibParamsVector 

# sums: 0=ET, 1=HT, 2=MET, 3=MHT
caloStage2Params.etSumLsb                = cms.double(0.5)
caloStage2Params.etSumEtaMin             = cms.vint32(1, 1, 1, 1)
caloStage2Params.etSumEtaMax             = cms.vint32(28,  28,  28,  28)
caloStage2Params.etSumEtThreshold        = cms.vdouble(0.,  35.,  0.,  35.)

# Layer 1 LUT specification
#
# Et-dependent scale factors
# ECal/HCal scale factors will be a 9*28 array:
#   28 eta scale factors (1-28)
#   in 9 ET bins (10, 15, 20, 25, 30, 35, 40, 45, Max)
#  So, index = etBin*28+ieta
caloStage2Params.layer1ECalScaleETBins = cms.vint32([1])
caloStage2Params.layer1ECalScaleFactors = cms.vdouble([1.]*28)
caloStage2Params.layer1HCalScaleETBins = cms.vint32([1])
caloStage2Params.layer1HCalScaleFactors = cms.vdouble([1.]*28)
caloStage2Params.layer1HFScaleETBins = cms.vint32([1])
caloStage2Params.layer1HFScaleFactors = cms.vdouble([1.]*12)
