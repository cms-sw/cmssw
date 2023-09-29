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
    egIsoLUTFile               = "L1Trigger/L1TCalorimeter/data/eg_IsoLUT_tight_Opt_L281_7p5_0p9_30p0_12_Jul_2022.txt",
    egIsoLUTFile2              = "L1Trigger/L1TCalorimeter/data/eg_IsoLUT_loose_Opt_L137_5p0_0p9_30p0_12_Jul_2022.txt",

    egIsoVetoNrTowersPhi       = 2,
    egPUSParams                = cms.vdouble(1,4,32), #Isolation window in firmware goes up to abs(ieta)=32 for now
    egCalibrationType          = "compressed",
    egCalibrationVersion       = 0,
    egCalibrationLUTFile       = "L1Trigger/L1TCalorimeter/data/egRecalibratedLUTv1_2022_v0_2.txt",

    # Tau
    isoTauEtaMax               = 25,
    tauSeedThreshold           = 0.,
    tauIsoLUTFile              = "L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_2022_calibThr1p7_rate14kHz_V11gs_effMin0p9_G3.txt",
    tauIsoLUTFile2             = "L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_2022_calibThr1p7_rate14kHz_V11gs_effMin0p9_G3.txt",
    tauCalibrationLUTFile      = "L1Trigger/L1TCalorimeter/data/Tau_Cal_LUT_2022_calibThr1p7_V11.txt",
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
    jetCalibrationLUTFile      = "L1Trigger/L1TCalorimeter/data/lut_calib_2022v5_ECALZS_HI_noHFJEC.txt",


    # sums: 0=ET, 1=HT, 2=MET, 3=MHT
    etSumEtaMin             = [1, 1, 1, 1, 1],
    etSumEtaMax             = [28,  26, 28,  26, 28],
    etSumEtThreshold        = [0.,  30.,  0.,  30., 0.], # only 2nd (HT) and 4th (MHT) values applied
    etSumMetPUSType         = "LUT", # et threshold from this LUT supercedes et threshold in line above
    etSumBypassEttPUS       = 1,
    etSumBypassEcalSumPUS   = 1,

    etSumMetPUSLUTFile               = "L1Trigger/L1TCalorimeter/data/metPumLUT_2022_HCALOff_p5.txt",

    etSumCentralityUpper = [2.0, 6.0, 53.5, 190.0, 645.0, 6000.0, 6000.0, 65535.0],
    etSumCentralityLower = [0.0,  1.5,  4.0, 47.5, 157.5, 4707.0, 4857.5, 65535.0],

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

    layer1HCalScaleETBins = cms.vint32([6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256]),
    layer1HCalScaleFactors = cms.vdouble([
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
