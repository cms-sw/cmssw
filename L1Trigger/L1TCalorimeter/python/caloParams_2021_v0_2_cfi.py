import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloParams_cfi import caloParamsSource
import L1Trigger.L1TCalorimeter.caloParams_cfi
caloStage2Params = L1Trigger.L1TCalorimeter.caloParams_cfi.caloParams.clone(

    # towers
    #towerLsbH        = 0.5
    #towerLsbE        = 0.5
    #towerLsbSum      = 0.5
    #towerNBitsH      = 8
    #towerNBitsE      = 8
    #towerNBitsSum    = 9
    #towerNBitsRatio  = 3
    #towerEncoding    = True

    # regions
    #regionLsb        = 0.5
    #regionPUSType    = "None"
    #regionPUSParams  = []

    # EG
    #egEtaCut                   = 28
    #egLsb                      = 0.5
    #egSeedThreshold            = 2.
    #egNeighbourThreshold       = 1.
    egHcalThreshold            = 0.,
    egTrimmingLUTFile          = "L1Trigger/L1TCalorimeter/data/egTrimmingLUT_10_v16.01.19.txt",
    #egMaxHcalEt                = 0.
    #egMaxPtHOverE              = 128.
    egHOverEcutBarrel          = 3,
    egHOverEcutEndcap          = 4,
    egBypassExtHOverE          = 0,
    egMaxHOverELUTFile         = "L1Trigger/L1TCalorimeter/data/HoverEIdentification_0.995_v15.12.23.txt",
    egCompressShapesLUTFile    = "L1Trigger/L1TCalorimeter/data/egCompressLUT_v4.txt",
    egShapeIdType              = "compressed",
    #egShapeIdVersion           = 0
    egShapeIdLUTFile           = "L1Trigger/L1TCalorimeter/data/shapeIdentification_adapt0.99_compressedieta_compressedE_compressedshape_v15.12.08.txt", #Not used any more in the current emulator version, merged with calibration LUT

    #egPUSType                  = "None"
    egIsolationType            = "compressed",
    egIsoLUTFile               = "L1Trigger/L1TCalorimeter/data/EG_Iso_LUT_04_04_2017.2.txt",
    egIsoLUTFile2              = "L1Trigger/L1TCalorimeter/data/EG_LoosestIso_2018.2.txt",
    #egIsoAreaNrTowersEta       = 2
    #egIsoAreaNrTowersPhi       = 4
    egIsoVetoNrTowersPhi       = 2,
    #egIsoPUEstTowerGranularity = cms.uint32(1)
    #egIsoMaxEtaAbsForTowerSum  = cms.uint32(4)
    #egIsoMaxEtaAbsForIsoSum    = cms.uint32(27)
    egPUSParams                = cms.vdouble(1,4,32), #Isolation window in firmware goes up to abs(ieta)=32 for now
    egCalibrationType          = "compressed",
    egCalibrationVersion       = 0,
    egCalibrationLUTFile       = "L1Trigger/L1TCalorimeter/data/corrections_Trimming10_compressedieta_compressedE_compressedshape_PANTELIS_v2_NEW_CALIBRATIONS_withShape_v17.04.04.txt",

    # Tau
    #tauLsb                     = 0.5
    isoTauEtaMax               = 25,
    tauSeedThreshold           = 0.,
    #tauNeighbourThreshold      = 0.
    #tauIsoAreaNrTowersEta      = 2
    #tauIsoAreaNrTowersPhi      = 4
    #tauIsoVetoNrTowersPhi      = 2
    #tauPUSType                 = "None"
    tauIsoLUTFile              = "L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_31_extrap_2018_FW_v10.0.0.txt",
    tauIsoLUTFile2             = "L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_Option_31_extrap_2018_FW_v10.0.0.txt",
    tauCalibrationLUTFile      = "L1Trigger/L1TCalorimeter/data/Tau_Calibration_LUT_2018_Layer1CalibrationNewHCAL_FW_v13.0.0.txt",
    tauCompressLUTFile         = "L1Trigger/L1TCalorimeter/data/tauCompressAllLUT_12bit_v3.txt",
    tauPUSParams               = [1,4,32],

    # jets
    #jetLsb                    = 0.5
    jetSeedThreshold           = 4.0,
    #jetNeighbourThreshold = 0.
    jetPUSType                 = "ChunkyDonut",
    #jetBypassPUS          = 0

    # Calibration options
    jetCalibrationType         = "LUT",
    jetCompressPtLUTFile       = "L1Trigger/L1TCalorimeter/data/lut_pt_compress_2017v1.txt",
    jetCompressEtaLUTFile      = "L1Trigger/L1TCalorimeter/data/lut_eta_compress_2017v1.txt",
    jetCalibrationLUTFile      = "L1Trigger/L1TCalorimeter/data/lut_calib_2018v1_ECALZS_noHFJEC.txt",


    # sums: 0=ET, 1=HT, 2=MET, 3=MHT
    #etSumLsb                = 0.5
    etSumEtaMin             = [1, 1, 1, 1, 1],
    etSumEtaMax             = [28,  26, 28,  26, 28],
    etSumEtThreshold        = [0.,  30.,  0.,  30., 0.], # only 2nd (HT) and 4th (MHT) values applied
    etSumMetPUSType         = "LUT", # et threshold from this LUT supercedes et threshold in line above
    #etSumEttPUSType         = "None"
    #etSumEcalSumPUSType     = "None"
    #etSumBypassMetPUS       = 0
    etSumBypassEttPUS       = 1,
    etSumBypassEcalSumPUS   = 1,
    #etSumXCalibrationType   = "None"
    #etSumYCalibrationType   = "None"
    #etSumEttCalibrationType = "None"
    #etSumEcalSumCalibrationType = "None"

    etSumMetPUSLUTFile               = "L1Trigger/L1TCalorimeter/data/newSFHBHEOnp5_METPUM_211124.txt",
    #etSumEttPUSLUTFile               = "L1Trigger/L1TCalorimeter/data/lut_towEtThresh_dummy.txt"
    #etSumEcalSumPUSLUTFile           = "L1Trigger/L1TCalorimeter/data/lut_towEtThresh_dummy.txt"
    #etSumXCalibrationLUTFile         = "L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt"
    #etSumYCalibrationLUTFile         = "L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt"
    #etSumEttCalibrationLUTFile       = "L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt"
    #etSumEcalSumCalibrationLUTFile   = "L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt"


    # Layer 1 SF
    layer1ECalScaleETBins = cms.vint32([3, 6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256]),
    layer1ECalScaleFactors = cms.vdouble([
        1.13, 1.13, 1.13, 1.12, 1.12, 1.12, 1.12, 1.12, 1.13, 1.12, 1.13, 1.13, 1.13, 1.14, 1.14, 1.13, 1.13, 1.31, 1.15, 1.27, 1.28, 1.31, 1.31, 1.32, 1.35, 0.00, 0.00, 0.00, 

        1.13, 1.13, 1.13, 1.12, 1.12, 1.12, 1.12, 1.12, 1.13, 1.12, 1.13, 1.13, 1.13, 1.14, 1.14, 1.13, 1.13, 1.31, 1.15, 1.27, 1.28, 1.31, 1.31, 1.32, 1.35, 1.38, 0.00, 0.00,

        1.08, 1.08, 1.09, 1.08, 1.09, 1.08, 1.08, 1.09, 1.10, 1.09, 1.09, 1.10, 1.09, 1.09, 1.09, 1.10, 1.10, 1.26, 1.11, 1.21, 1.20, 1.23, 1.25, 1.28, 1.31, 1.33, 1.21, 0.00, 

        1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.07, 1.07, 1.06, 1.07, 1.07, 1.07, 1.07, 1.08, 1.08, 1.07, 1.08, 1.19, 1.09, 1.16, 1.16, 1.20, 1.22, 1.23, 1.28, 1.29, 1.18, 1.09, 

        1.04, 1.04, 1.04, 1.05, 1.05, 1.04, 1.05, 1.05, 1.05, 1.06, 1.05, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.16, 1.08, 1.15, 1.15, 1.19, 1.20, 1.22, 1.26, 1.31, 1.15, 1.08, 

        1.04, 1.03, 1.04, 1.04, 1.03, 1.03, 1.04, 1.04, 1.04, 1.04, 1.04, 1.05, 1.05, 1.06, 1.05, 1.05, 1.05, 1.14, 1.07, 1.12, 1.14, 1.17, 1.18, 1.21, 1.25, 1.27, 1.15, 1.07, 

        1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.04, 1.03, 1.04, 1.03, 1.03, 1.03, 1.04, 1.04, 1.05, 1.05, 1.03, 1.13, 1.06, 1.11, 1.13, 1.15, 1.17, 1.20, 1.23, 1.25, 1.12, 1.08, 

        1.03, 1.02, 1.03, 1.02, 1.03, 1.00, 1.03, 1.03, 1.03, 1.03, 1.02, 1.03, 1.04, 1.04, 1.04, 1.04, 1.02, 1.11, 1.05, 1.11, 1.12, 1.14, 1.16, 1.18, 1.22, 1.26, 1.08, 1.05, 

        1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.04, 1.03, 1.03, 1.04, 1.11, 1.05, 1.11, 1.11, 1.14, 1.17, 1.17, 1.21, 1.22, 1.07, 1.05, 

        1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.09, 1.05, 1.10, 1.11, 1.13, 1.15, 1.17, 1.20, 1.21, 1.06, 1.05, 

        1.01, 1.02, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.09, 1.05, 1.09, 1.10, 1.12, 1.14, 1.16, 1.20, 1.23, 1.07, 1.05, 

        1.00, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.02, 1.03, 1.02, 1.03, 1.03, 1.08, 1.05, 1.10, 1.09, 1.12, 1.13, 1.17, 1.19, 1.23, 1.04, 1.03, 

        1.00, 1.01, 1.01, 1.01, 1.01, 1.01, 1.00, 1.01, 1.01, 1.02, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.02, 1.06, 1.05, 1.09, 1.09, 1.11, 1.12, 1.16, 1.18, 1.22, 1.00, 1.03, 

        1.00, 1.00, 1.00, 1.01, 1.01, 1.01, 1.00, 1.01, 1.01, 1.02, 1.00, 1.01, 1.02, 1.02, 1.02, 1.02, 1.02, 1.06, 1.04, 1.08, 1.09, 1.10, 1.13, 1.15, 1.18, 1.21, 1.00, 1.03
    ]),

    layer1HCalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256]),
    layer1HCalScaleFactors = cms.vdouble([
        1.55, 1.59, 1.60, 1.60, 1.58, 1.62, 1.63, 1.63, 1.63, 1.65, 1.65, 1.71, 1.69, 1.72, 1.84, 1.98, 1.98, 1.51, 1.55, 1.56, 1.42, 1.44, 1.46, 1.46, 1.51, 1.44, 1.29, 1.23, 

        1.39, 1.39, 1.40, 1.42, 1.40, 1.42, 1.45, 1.43, 1.43, 1.45, 1.47, 1.49, 1.47, 1.51, 1.57, 1.67, 1.70, 1.32, 1.35, 1.36, 1.24, 1.26, 1.27, 1.30, 1.32, 1.31, 1.16, 1.10, 

        1.31, 1.33, 1.33, 1.34, 1.33, 1.34, 1.35, 1.37, 1.36, 1.37, 1.39, 1.39, 1.39, 1.39, 1.45, 1.54, 1.57, 1.22, 1.25, 1.27, 1.16, 1.19, 1.20, 1.22, 1.25, 1.24, 1.10, 1.05, 

        1.27, 1.28, 1.29, 1.29, 1.29, 1.28, 1.31, 1.31, 1.30, 1.31, 1.33, 1.34, 1.33, 1.34, 1.41, 1.46, 1.48, 1.19, 1.20, 1.20, 1.12, 1.13, 1.15, 1.17, 1.20, 1.20, 1.06, 1.01, 

        1.22, 1.22, 1.23, 1.23, 1.23, 1.24, 1.24, 1.26, 1.25, 1.27, 1.27, 1.28, 1.28, 1.27, 1.32, 1.38, 1.41, 1.12, 1.15, 1.16, 1.08, 1.10, 1.11, 1.13, 1.15, 1.15, 1.03, 0.98, 

        1.17, 1.19, 1.17, 1.19, 1.19, 1.19, 1.20, 1.22, 1.20, 1.21, 1.21, 1.22, 1.22, 1.23, 1.26, 1.31, 1.33, 1.10, 1.10, 1.10, 1.04, 1.06, 1.07, 1.09, 1.11, 1.10, 0.99, 0.95, 

        1.14, 1.15, 1.14, 1.15, 1.16, 1.15, 1.16, 1.17, 1.16, 1.17, 1.19, 1.18, 1.18, 1.19, 1.22, 1.26, 1.26, 1.06, 1.07, 1.08, 1.02, 1.03, 1.04, 1.06, 1.07, 1.07, 0.96, 0.92, 

        1.11, 1.11, 1.13, 1.12, 1.11, 1.13, 1.13, 1.13, 1.12, 1.14, 1.15, 1.15, 1.14, 1.15, 1.17, 1.20, 1.23, 1.03, 1.05, 1.05, 1.00, 1.01, 1.02, 1.03, 1.05, 1.03, 0.95, 0.91, 

        1.08, 1.09, 1.09, 1.08, 1.09, 1.10, 1.10, 1.11, 1.11, 1.11, 1.12, 1.11, 1.11, 1.12, 1.13, 1.17, 1.16, 1.01, 1.02, 1.03, 0.98, 0.99, 0.99, 1.01, 1.02, 1.01, 0.94, 0.89, 

        1.06, 1.07, 1.06, 1.07, 1.07, 1.07, 1.08, 1.08, 1.07, 1.07, 1.08, 1.08, 1.08, 1.09, 1.10, 1.14, 1.13, 1.00, 1.02, 1.02, 0.97, 0.98, 0.98, 0.99, 1.00, 1.00, 0.92, 0.87, 

        1.03, 1.04, 1.04, 1.04, 1.04, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.06, 1.06, 1.09, 1.09, 0.97, 0.99, 1.00, 0.95, 0.96, 0.96, 0.97, 0.99, 0.98, 0.90, 0.85, 

        1.00, 1.00, 1.00, 1.01, 1.01, 1.01, 1.02, 1.02, 1.01, 1.01, 1.02, 1.02, 1.01, 0.98, 1.01, 1.02, 1.02, 0.96, 0.97, 1.00, 0.93, 0.94, 0.94, 0.95, 0.96, 0.96, 0.89, 0.82, 

        0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.95, 0.95, 0.95, 0.96, 0.95, 0.93, 0.95, 0.95, 0.93, 0.93, 0.94, 0.94, 0.95, 0.95, 0.88, 0.82
    ]),

    layer1HFScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256]),

    layer1HFScaleFactors = cms.vdouble([
        1.35, 1.09, 1.12, 1.10, 1.17, 1.18, 1.19, 1.23, 1.25, 1.32, 1.61, 1.79, 

        1.27, 1.01, 1.09, 1.03, 1.04, 1.05, 1.09, 1.11, 1.18, 1.19, 1.48, 1.67, 

        1.15, 0.98, 1.05, 1.02, 1.00, 0.99, 1.03, 1.04, 1.10, 1.12, 1.39, 1.66, 

        1.14, 0.96, 1.03, 0.97, 0.96, 0.96, 0.98, 1.00, 1.04, 1.07, 1.35, 1.59, 

        1.07, 0.97, 1.00, 0.96, 0.91, 0.92, 0.95, 0.96, 1.01, 1.03, 1.28, 1.56, 

        1.03, 0.94, 0.97, 0.94, 0.88, 0.90, 0.92, 0.94, 0.98, 1.01, 1.27, 1.53, 

        1.01, 0.92, 0.96, 0.90, 0.87, 0.89, 0.91, 0.93, 0.96, 0.99, 1.23, 1.48, 

        0.98, 0.89, 0.96, 0.87, 0.86, 0.87, 0.89, 0.91, 0.94, 0.97, 1.19, 1.47, 

        0.95, 0.88, 0.94, 0.87, 0.86, 0.86, 0.88, 0.90, 0.94, 0.96, 1.16, 1.43, 

        0.93, 0.88, 0.93, 0.87, 0.86, 0.87, 0.88, 0.90, 0.93, 0.95, 1.14, 1.42, 

        0.92, 0.86, 0.90, 0.86, 0.85, 0.86, 0.88, 0.89, 0.92, 0.95, 1.12, 1.41, 

        0.90, 0.85, 0.90, 0.85, 0.84, 0.86, 0.88, 0.90, 0.93, 0.95, 1.09, 1.35, 

        0.86, 0.85, 0.89, 0.85, 0.85, 0.86, 0.88, 0.90, 0.93, 0.95, 1.10, 1.27
    ])
)

