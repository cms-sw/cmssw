import FWCore.ParameterSet.Config as cms

l1tCaloParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TCaloParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

l1tCaloStage2Params = cms.ESProducer(
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
    egLsb        = cms.double(0.5),
    egSeedThreshold      = cms.double(2.),
    egNeighbourThreshold = cms.double(1.),
    egMaxHcalEt          = cms.double(0.),
    egEtToRemoveHECut    = cms.double(128.),
    egMaxHOverE          = cms.double(0.15),  
    egIsoPUSType         = cms.string("None"),
    egIsoLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT_PU40bx25.txt"),
    egIsoAreaNrTowersEta = cms.uint32(2),
    egIsoAreaNrTowersPhi = cms.uint32(4),
    egIsoVetoNrTowersPhi = cms.uint32(3),
    egIsoPUEstTowerGranularity = cms.uint32(1),
    egIsoMaxEtaAbsForTowerSum = cms.uint32(4),
    egIsoMaxEtaAbsForIsoSum = cms.uint32(27),

    # Tau
    tauLsb                = cms.double(0.5),
    tauSeedThreshold      = cms.double(0.),
    tauNeighbourThreshold = cms.double(0.),
    tauIsoPUSType         = cms.string("None"),
    tauIsoLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauIsoLUT.txt"),

    # jets
    jetLsb                = cms.double(0.5),
    jetSeedThreshold      = cms.double(0.),
    jetNeighbourThreshold = cms.double(0.),
    jetPUSType            = cms.string("None"),
    jetCalibrationType    = cms.string("None"),
    jetCalibrationParams  = cms.vdouble(),

    # sums
    etSumLsb                = cms.double(0.5),
    ettEtaMin             = cms.int32(-999),
    ettEtaMax             = cms.int32(999),
    ettEtThreshold        = cms.double(0.),
    httEtaMin             = cms.int32(-999),
    httEtaMax             = cms.int32(999),
    httEtThreshold        = cms.double(0.),
    metEtaMin             = cms.int32(-999),
    metEtaMax             = cms.int32(999),
    metEtThreshold        = cms.double(0.),
    mhtEtaMin             = cms.int32(-999),
    mhtEtaMax             = cms.int32(999),
    mhtEtThreshold        = cms.double(0.)

)
