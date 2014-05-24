import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.regionSF_cfi import *
from L1Trigger.L1TCalorimeter.jetSF_cfi import *


l1tCaloParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TCaloParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

l1tCaloParams = cms.ESProducer(
    "l1t::CaloParamsESProducer",

    # towers
    towerLsbH        = cms.double(0.5),    #not used by Stage1
    towerLsbE        = cms.double(0.5),    #not used by Stage1
    towerLsbSum      = cms.double(0.5),    #not used by Stage1
    towerNBitsH      = cms.int32(8),    #not used by Stage1
    towerNBitsE      = cms.int32(8),    #not used by Stage1
    towerNBitsSum    = cms.int32(9),    #not used by Stage1
    towerNBitsRatio  = cms.int32(3),    #not used by Stage1
    towerEncoding    = cms.bool(False),    #not used by Stage1

    # regions
    regionLsb        = cms.double(0.5),       #not used by Stage1
    regionPUSType    = cms.string("PUM0"),       #"None" for no PU subtraction, "PUM0"
    regionPUSParams  = regionSubtraction_PU40_MC13TeV,

    # EG
    egLsb                = cms.double(1.0),
    egSeedThreshold      = cms.double(1.),
    egNeighbourThreshold = cms.double(1.),      #not used by Stage1
    egMaxHcalEt          = cms.double(0.),    #not used by Stage1
    egEtToRemoveHECut    = cms.double(128.),    #not used by Stage1
    egMaxHOverE          = cms.double(0.15),    #not used by Stage1
    egIsoPUSType         = cms.string("None"),    #not used by Stage1
    egIsoLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/egIsoLUT.txt"),    #not used by Stage1
    egIsoAreaNrTowersEta = cms.uint32(2),    #not used by Stage1
    egIsoAreaNrTowersPhi = cms.uint32(4),    #not used by Stage1
    egIsoVetoNrTowersPhi = cms.uint32(3),    #not used by Stage1
    egIsoPUEstTowerGranularity = cms.uint32(1),    #not used by Stage1
    egIsoMaxEtaAbsForTowerSum = cms.uint32(4),    #not used by Stage1
    egIsoMaxEtaAbsForIsoSum = cms.uint32(27),    #not used by Stage1
    
    # Tau
    tauLsb                = cms.double(0.5),    #not used by Stage1
    tauSeedThreshold      = cms.double(7.),    #not used by Stage1
    tauNeighbourThreshold = cms.double(0.),    #not used by Stage1
    tauIsoPUSType         = cms.string("None"),    #not used by Stage1
    tauIsoLUTFile         = cms.FileInPath("L1Trigger/L1TCalorimeter/data/tauIsoLUT.txt"),    #not used by Stage1

    # jets
    jetLsb                = cms.double(0.5),
    jetSeedThreshold      = cms.double(10.),
    jetNeighbourThreshold = cms.double(0.),
    jetPUSType            = cms.string("None"),    #not used by Stage1
    jetPUSParams          = cms.vdouble(),    #not used by Stage1
    jetCalibrationType    = cms.string("Stage1JEC"),  #"None" for no calibration, "Stage1JEC"
    jetCalibrationParams  = jetSF_8TeV_data,

    # sums
    etSumLsb                = cms.double(0.5),    #not used by Stage1
    ettEtaMin             = cms.int32(4),    #not used by Stage1
    ettEtaMax             = cms.int32(17),    #not used by Stage1
    ettEtThreshold        = cms.double(0.),    #not used by Stage1
    httEtaMin             = cms.int32(4),    #not used by Stage1
    httEtaMax             = cms.int32(17),    #not used by Stage1
    httEtThreshold        = cms.double(7.),    #not used by Stage1
    metEtaMin             = cms.int32(4),    #not used by Stage1
    metEtaMax             = cms.int32(17),    #not used by Stage1
    metEtThreshold        = cms.double(0.),    #not used by Stage1
    mhtEtaMin             = cms.int32(4),    #not used by Stage1
    mhtEtaMax             = cms.int32(17),    #not used by Stage1
    mhtEtThreshold        = cms.double(0.)    #not used by Stage1

)
