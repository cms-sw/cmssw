import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.GctConfigProducers.L1GctConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff import *
from L1TriggerConfig.L1GeometryProducers.l1CaloGeomConfig_cff import *
gctemu = cms.EDFilter("L1GctTest",
    doFirmware = cms.untracked.bool(False),
    doElectrons = cms.untracked.bool(False),
    doEnergyAlgos = cms.untracked.bool(False),
    inputFile = cms.untracked.string(''),
    energySumsFile = cms.untracked.string(''),
    referenceFile = cms.untracked.string(''),
    jetThresholdForHtSumGeV = cms.double(5.0),
    preSamples = cms.uint32(2),
    postSamples = cms.uint32(2)
)

