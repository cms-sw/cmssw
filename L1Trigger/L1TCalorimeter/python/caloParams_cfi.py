import FWCore.ParameterSet.Config as cms

caloParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TCaloParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

caloParams = cms.ESProducer(
    "L1TCaloParamsESProducer",

    # towers
    towerLsbH        = cms.double(0.5),
    towerLsbE        = cms.double(0.5),
    towerLsbSum      = cms.double(0.5),
    towerNBitsH      = cms.int32(8),
    towerNBitsE      = cms.int32(8),
    towerNBitsSum    = cms.int32(9),
    towerNBitsRatio  = cms.int32(3),
    towerEncoding    = cms.bool(True),

    # regions
    regionLsb        = cms.double(0.5),
    regionPUSType    = cms.string("None"),
    regionPUSVersion = cms.int32(0),
    regionPUSParams  = cms.vdouble(),

    # EG
    egEtaCut                   = cms.int32(0),
    egLsb                      = cms.double(0.5),
    egSeedThreshold            = cms.double(2.),
    egNeighbourThreshold       = cms.double(1.),
    egHcalThreshold            = cms.double(1.),
    egMaxHcalEt                = cms.double(0.),
    egTrimmingLUTFile          = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egTrimmingLUT_corners.txt"),
    egMaxPtHOverE          = cms.double(128.),
    egMaxHOverE                = cms.double(0.15),
    egMaxHOverELUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egMaxHOverELUT.txt"),
    egCompressShapesLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCompressShapesLUT.txt"),
    egShapeIdType              = cms.string("unspecified"),
    egShapeIdVersion           = cms.uint32(0),
    egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egShapeIdLUT.txt"),
    egCalibrationType          = cms.string("unspecified"),
    egCalibrationVersion       = cms.uint32(0),
    egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCalibrationLUT.txt"),
    egMinPtJetIsolation      = cms.int32(25),
    egMaxPtJetIsolation      = cms.int32(63),
    egMinPtHOverEIsolation                    = cms.int32(1),
    egMaxPtHOverEIsolation                    = cms.int32(40),
    egPUSType               = cms.string("None"),
    egIsolationType          = cms.string("unspecified"),
    egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT.txt"),
    #egIsoLUTFileBarrel         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT_stage1_isol0.30.txt"),
    #egIsoLUTFileEndcaps        = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT_stage1_isol0.50.txt"),
    egIsoAreaNrTowersEta       = cms.uint32(2),
    egIsoAreaNrTowersPhi       = cms.uint32(4),
    egIsoVetoNrTowersPhi       = cms.uint32(3),
    egIsoPUEstTowerGranularity = cms.uint32(1),
    egIsoMaxEtaAbsForTowerSum  = cms.uint32(4),
    egIsoMaxEtaAbsForIsoSum    = cms.uint32(27),

    # Tau
    tauRegionMask                 = cms.int32(0),
    tauLsb                        = cms.double(0.5),
    tauSeedThreshold              = cms.double(7.),
    tauNeighbourThreshold         = cms.double(0.),
    tauMaxPtTauVeto              = cms.double(64.),
    tauMinPtJetIsolationB               = cms.double(192.),
    tauMaxJetIsolationB  = cms.double(100.),
    tauMaxJetIsolationA    = cms.double(0.1),
    tauPUSType                 = cms.string("None"),
    isoTauEtaMin                  = cms.int32(0),
    isoTauEtaMax                  = cms.int32(17),
    tauIsoAreaNrTowersEta 		  = cms.uint32(2),
    tauIsoAreaNrTowersPhi		  = cms.uint32(4),
    tauIsoVetoNrTowersPhi 		  = cms.uint32(2),
    tauIsoLUTFile                 = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauIsoLUTetPU.txt"),
    tauCalibrationLUTFileEta      = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTEta.txt"),
    tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUT.txt"),
    tauCompressLUTFile            = cms.FileInPath("L1Trigger/L1TCalorimeter/data/Tau_Eta_Et_compression_LUT.txt"),
    tauEtToHFRingEtLUTFile        = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauHwEtToHFRingScale_LUT.txt"),
    tauPUSParams                  = cms.vdouble(1,4,27),

    # jets
    jetRegionMask            = cms.int32(0),
    jetLsb                   = cms.double(0.5),
    jetSeedThreshold         = cms.double(0.),
    jetNeighbourThreshold    = cms.double(0.),
    jetPUSType               = cms.string("None"),
    jetCalibrationType       = cms.string("None"),
    jetCalibrationParams     = cms.vdouble(),
    jetCompressPtLUTFile     = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_pt_compress.txt"),
    jetCompressEtaLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_eta_compress.txt"),
    jetCalibrationLUTFile    = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_add_mult.txt"),

    # sums
    etSumLsb                 = cms.double(0.5),
    etSumEtaMin              = cms.vint32(-999, -999, -999, -999),
    etSumEtaMax              = cms.vint32(999,  999,  999,  999),
    etSumEtThreshold         = cms.vdouble(0.,  0.,   0.,   0.),
    etSumXPUSLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt"),
    etSumYPUSLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt"),
    etSumEttPUSLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt"),
    etSumEcalSumPUSLUTFile   = cms.FileInPath("L1Trigger/L1TCalorimeter/data/lut_etSumPUS_dummy.txt"),


    # HI
    centralityNodeVersion = cms.int32(1),
    centralityRegionMask = cms.int32(0),
    minimumBiasThresholds = cms.vint32(0, 0, 0, 0),
    centralityLUTFile = cms.FileInPath("L1Trigger/L1TCalorimeter/data/centralityLUT_stage1.txt"),
    q2LUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/q2LUT_stage1.txt")

)
