import FWCore.ParameterSet.Config as cms

caloParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TCaloParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

caloParams = cms.ESProducer(
    "l1t::CaloParamsESProducer",

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
    egEtToRemoveHECut          = cms.double(128.),
    egMaxHOverE                = cms.double(0.15),
    egMaxHOverELUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egMaxHOverELUT.txt"),
    egShapeIdLUTFile           = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egShapeIdLUT.txt"),
    egCalibrationLUTFile       = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egCalibrationLUT.txt"),
    egRelativeJetIsolationBarrelCut  = cms.double(0.3),  # 0.3 for loose, 0.2 for tight
    egRelativeJetIsolationEndcapCut  = cms.double(0.5),  # 0.5 for loose, 0.4 for tight
    egIsoPUSType               = cms.string("None"),
    egIsoLUTFile               = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT.txt"),
    egIsoAreaNrTowersEta       = cms.uint32(2),
    egIsoAreaNrTowersPhi       = cms.uint32(4),
    egIsoVetoNrTowersPhi       = cms.uint32(3),
    egIsoPUEstTowerGranularity = cms.uint32(1),
    egIsoMaxEtaAbsForTowerSum  = cms.uint32(4),
    egIsoMaxEtaAbsForIsoSum    = cms.uint32(27),
    
    # Tau
    tauLsb                        = cms.double(0.5),
    tauSeedThreshold              = cms.double(0.),
    tauNeighbourThreshold         = cms.double(0.),
    switchOffTauVeto              = cms.double(70.),
    switchOffTauIso               = cms.double(60.),
    tauRelativeJetIsolationLimit  = cms.double(100.),
    tauRelativeJetIsolationCut    = cms.double(0.15),
    tauIsoPUSType                 = cms.string("None"),
    tauIsoLUTFile                 = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauIsoLUT.txt"),
    tauCalibrationLUTFileBarrelA  = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTBarrelA.txt"),
    tauCalibrationLUTFileBarrelB  = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTBarrelB.txt"),
    tauCalibrationLUTFileBarrelC  = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTBarrelC.txt"),
    tauCalibrationLUTFileEndcapsA = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTEndcapsA.txt"),
    tauCalibrationLUTFileEndcapsB = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTEndcapsB.txt"),
    tauCalibrationLUTFileEndcapsC = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTEndcapsC.txt"),
    tauCalibrationLUTFileEta      = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauCalibrationLUTEta.txt"),


    # jets
    jetLsb                = cms.double(0.5),
    jetSeedThreshold      = cms.double(0.),
    jetNeighbourThreshold = cms.double(0.),
    jetPUSType            = cms.string("None"),
    jetPUSParams          = cms.vdouble(),
    jetCalibrationType    = cms.string("None"),
    jetCalibrationParams  = cms.vdouble(),

    # sums
    etSumLsb                = cms.double(0.5),
    etSumEtaMin             = cms.vint32(-999, -999, -999, -999),
    etSumEtaMax             = cms.vint32(999,  999,  999,  999),
    etSumEtThreshold        = cms.vdouble(0.,  0.,   0.,   0.)

)
