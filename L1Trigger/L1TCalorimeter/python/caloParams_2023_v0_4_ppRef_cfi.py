import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloParams_cfi import caloParamsSource
import L1Trigger.L1TCalorimeter.caloParams_cfi
caloStage2Params = L1Trigger.L1TCalorimeter.caloParams_cfi.caloParams.clone(

    # EG
    egEtaCut                   = 24,
    egHcalThreshold            = 0.,
    egTrimmingLUTFile          = "L1Trigger/L1TCalorimeter/data/egTrimmingLUT_10_v16.01.19.txt",
    egHOverEcutBarrel          = 1,
    egHOverEcutEndcap          = 1,
    egBypassExtHOverE          = 1,
    egBypassShape              = 1,
    egBypassECALFG             = 1,
    egMaxHOverELUTFile         = "L1Trigger/L1TCalorimeter/data/HoverEIdentification_0.995_v15.12.23.txt",
    egCompressShapesLUTFile    = "L1Trigger/L1TCalorimeter/data/egCompressLUT_v4.txt",
    egShapeIdType              = "compressed",
    egShapeIdLUTFile           = "L1Trigger/L1TCalorimeter/data/shapeIdentification_adapt0.99_compressedieta_compressedE_compressedshape_v15.12.08.txt", #Not used any more in the current emulator version, merged with calibration LUT

    egIsolationType            = "compressed",
    egIsoLUTFile               = "L1Trigger/L1TCalorimeter/data/EG_Iso_LUT_Flat_WP_v2_Tight1358_20p0_0p7_40p0_v1_APR23.txt",
    egIsoLUTFile2              = "L1Trigger/L1TCalorimeter/data/EG_Iso_LUT_Flat_WP_v2_Loose610_10p0_0p7_40p0_v1_APR23.txt",

    egIsoVetoNrTowersPhi       = 2,
    egPUSParams                = cms.vdouble(1,4,32), #Isolation window in firmware goes up to abs(ieta)=32 for now
    egCalibrationType          = "compressed",
    egCalibrationVersion       = 0,
    egCalibrationLUTFile       = "L1Trigger/L1TCalorimeter/data/EG_Calibration_LUT_correctedEtCalibLUT_v1_APR2023.txt",

    # Tau
    isoTauEtaMax               = 25,
    tauSeedThreshold           = 0.,
    tauIsoLUTFile              = "L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_2023_calibThr1p7_V2gs_effMin0p9_eMin16_eMax60.txt",
    tauIsoLUTFile2             = "L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_2023_calibThr1p7_V2gs_effMin0p9_eMin16_eMax60.txt",
    tauCalibrationLUTFile      = "L1Trigger/L1TCalorimeter/data/Tau_Cal_LUT_2023_calibThr1p7_V2.txt",
    tauCompressLUTFile         = "L1Trigger/L1TCalorimeter/data/tauCompressAllLUT_12bit_v3.txt",
    tauPUSParams               = [1,4,32],

    # jets
    jetSeedThreshold           = 4.0,
    jetPUSType                 = "PhiRing1",
    jetPUSUsePhiRing           = 1, 

    # Calibration options
    jetCalibrationType         = "LUT",
    jetCompressPtLUTFile       = "L1Trigger/L1TCalorimeter/data/lut_pt_compress_2017v1.txt",
    jetCompressEtaLUTFile      = "L1Trigger/L1TCalorimeter/data/lut_eta_compress_2017v1.txt",
    jetCalibrationLUTFile      = "L1Trigger/L1TCalorimeter/data/lut_calib_2023v0_ECALZS_PhiRing.txt",


    # sums: 0=ET, 1=HT, 2=MET, 3=MHT
    etSumEtaMin             = [1, 1, 1, 1, 1],
    etSumEtaMax             = [28,  26, 28,  26, 28],
    etSumEtThreshold        = [0.,  30.,  0.,  30., 0.], # only 2nd (HT) and 4th (MHT) values applied
    etSumMetPUSType         = "LUT", # et threshold from this LUT supercedes et threshold in line above
    etSumBypassEttPUS       = 1,
    etSumBypassEcalSumPUS   = 1,

    etSumMetPUSLUTFile               = "L1Trigger/L1TCalorimeter/data/metPumLUT_2023v0_puppiMet_fit.txt",


    # Layer 1 SF
    layer1ECalScaleETBins = cms.vint32([3, 6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256]),
    layer1ECalScaleFactors = cms.vdouble([
        1.12, 1.13, 1.13, 1.12, 1.12, 1.12, 1.13, 1.12, 1.13, 1.12, 1.13, 1.13, 1.14, 1.13, 1.13, 1.13, 1.14, 1.26, 1.11, 1.20, 1.21, 1.22, 1.19, 1.20, 1.19, 0.00, 0.00, 0.00,
        1.12, 1.13, 1.13, 1.12, 1.12, 1.12, 1.13, 1.12, 1.13, 1.12, 1.13, 1.13, 1.14, 1.13, 1.13, 1.13, 1.14, 1.26, 1.11, 1.20, 1.21, 1.22, 1.19, 1.20, 1.19, 1.22, 0.00, 0.00,
        1.08, 1.09, 1.08, 1.08, 1.11, 1.08, 1.09, 1.09, 1.09, 1.09, 1.15, 1.09, 1.10, 1.10, 1.10, 1.10, 1.10, 1.23, 1.07, 1.15, 1.14, 1.16, 1.14, 1.14, 1.15, 1.14, 1.14, 0.00, 
        1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.07, 1.07, 1.07, 1.07, 1.07, 1.08, 1.07, 1.09, 1.08, 1.17, 1.06, 1.11, 1.10, 1.13, 1.10, 1.10, 1.11, 1.11, 1.11, 1.09, 
        1.04, 1.05, 1.04, 1.05, 1.04, 1.05, 1.06, 1.06, 1.05, 1.05, 1.05, 1.06, 1.06, 1.06, 1.06, 1.06, 1.07, 1.15, 1.04, 1.09, 1.09, 1.10, 1.09, 1.09, 1.10, 1.10, 1.10, 1.08, 
        1.04, 1.03, 1.04, 1.04, 1.04, 1.04, 1.04, 1.04, 1.04, 1.04, 1.04, 1.04, 1.05, 1.06, 1.04, 1.05, 1.05, 1.13, 1.03, 1.07, 1.08, 1.08, 1.08, 1.07, 1.07, 1.09, 1.08, 1.07, 
        1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.04, 1.04, 1.05, 1.05, 1.05, 1.05, 1.05, 1.12, 1.03, 1.06, 1.06, 1.08, 1.07, 1.07, 1.06, 1.08, 1.07, 1.06, 
        1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.04, 1.04, 1.04, 1.04, 1.04, 1.03, 1.10, 1.02, 1.05, 1.06, 1.06, 1.06, 1.06, 1.05, 1.06, 1.06, 1.06, 
        1.02, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.04, 1.03, 1.03, 1.02, 1.07, 1.02, 1.04, 1.04, 1.05, 1.06, 1.05, 1.05, 1.06, 1.06, 1.05, 
        1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.09, 1.02, 1.04, 1.05, 1.05, 1.05, 1.05, 1.04, 1.05, 1.06, 1.05, 
        1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.08, 1.01, 1.04, 1.04, 1.05, 1.05, 1.04, 1.04, 1.05, 1.06, 1.05, 
        1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.02, 1.01, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.06, 1.01, 1.04, 1.04, 1.05, 1.04, 1.03, 1.03, 1.04, 1.05, 1.04, 
        1.01, 1.00, 1.01, 1.01, 1.01, 1.01, 1.01, 1.00, 1.01, 1.02, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.03, 1.04, 1.01, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.00, 1.01, 
        1.02, 1.00, 1.00, 1.02, 1.00, 1.01, 1.01, 1.00, 1.00, 1.02, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.02, 1.04, 1.01, 1.03, 1.03, 1.03, 1.03, 1.02, 1.02, 1.02, 1.00, 1.01
    ]),

    layer1HCalScaleETBins = cms.vint32([1, 6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256]),
    layer1HCalScaleFactors = cms.vdouble([
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, # ZERO-SUPPRESS <1GeV (i.e. 0.5GeV) IN THE BARREL ONLY (ieta<=15 == eta<=1.305)
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00
    ]),

    layer1HFScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256]),
    layer1HFScaleFactors = cms.vdouble([
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00
    ]),

    # HCal FB LUT
    layer1HCalFBLUTUpper = cms.vuint32([
    0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 
    ]),

    layer1HCalFBLUTLower = cms.vuint32([
    0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 
    ])
)
