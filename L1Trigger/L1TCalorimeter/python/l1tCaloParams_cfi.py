import FWCore.ParameterSet.Config as cms

l1tCaloParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TCaloParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

l1tStage2CaloParams = cms.ESProducer(
    "l1t::L1TCaloParamsESProducer",

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
    regionPUSType    = cms.string("None"),
    regionPUSParams  = cms.vdouble(),

    # EG
    egSeedThreshold      = cms.double(0.),
    egNeighbourThreshold = cms.double(0.),
    egMaxHcalEt          = cms.double(0.),
    egMaxHOverE          = cms.double(20.),  #cut is H/E <= egMaxHOverE/128
    egIsoPUSType         = cms.string("None"),
    egIsoLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT.txt"),
    egIsoAreaNrTowersEta = cms.uint32(2),
    egIsoAreaNrTowersPhi = cms.uint32(4),
    egIsoVetoNrTowersPhi = cms.uint32(3),
    egIsoPUEstTowerGranularity = cms.uint32(1),
    egIsoMaxEtaAbsForTowerSum = cms.uint32(4),

    # Tau
    tauSeedThreshold      = cms.double(0.),
    tauNeighbourThreshold = cms.double(0.),
    tauIsoPUSType         = cms.string("None"),
    tauIsoLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauIsoLUT.txt"),

    # jets
    jetSeedThreshold      = cms.double(0.),
    jetNeighbourThreshold = cms.double(0.),
    jetPUSType            = cms.string("None"),
    jetCalibrationType    = cms.string("None"),
    jetCalibrationParams  = cms.vdouble(),

    # sums
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
