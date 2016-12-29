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
    towerEncoding    = cms.bool(False),

    # regions
    regionLsb        = cms.double(0.5),
    regionPUSType    = cms.string("None"),
    regionPUSParams  = cms.vdouble(),

    # EG
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
    egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egShapeIdLUT.txt"),
    egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCalibrationLUT.txt"),
    egMinPtJetIsolation      = cms.int32(25),
    egMaxPtJetIsolation      = cms.int32(63),
    egMinPtHOverEIsolation                    = cms.int32(1),
    egMaxPtHOverEIsolation                    = cms.int32(40),
    egPUSType               = cms.string("None"),
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
    tauCalibrationLUTFileBarrelA  = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTBarrelA.txt"),
    tauCalibrationLUTFileBarrelB  = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTBarrelB.txt"),
    tauCalibrationLUTFileBarrelC  = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTBarrelC.txt"),
    tauCalibrationLUTFileEndcapsA = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTEndcapsA.txt"),
    tauCalibrationLUTFileEndcapsB = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTEndcapsB.txt"),
    tauCalibrationLUTFileEndcapsC = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTEndcapsC.txt"),
    tauCalibrationLUTFileEta      = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTEta.txt"),
    tauCalibrationLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUT.txt"),
    tauPUSParams                  = cms.vdouble(1,4,27),


    # jets
    jetLsb                = cms.double(0.5),
    jetSeedThreshold      = cms.double(0.),
    jetNeighbourThreshold = cms.double(0.),
    jetPUSType            = cms.string("None"),
    jetPUSParams          = cms.vdouble(),
    jetCalibrationType    = cms.string("None"),
    jetCalibrationParams  = cms.vdouble(),
    jetCalibrationLUTFile = cms.FileInPath("L1Trigger/L1TCalorimeter/data/jetCalibrationLUT_stage1.txt"),

    # sums
    etSumLsb                = cms.double(0.5),
    etSumEtaMin             = cms.vint32(-999, -999, -999, -999),
    etSumEtaMax             = cms.vint32(999,  999,  999,  999),
    etSumEtThreshold        = cms.vdouble(0.,  0.,   0.,   0.),

    # HI
    centralityLUTFile = cms.FileInPath("L1Trigger/L1TCalorimeter/data/centralityLUT_stage1.txt"),
    q2LUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/q2LUT_stage1.txt")

)
