import FWCore.ParameterSet.Config as cms

GlobalParameters = cms.ESProducer("StableParametersTrivialProducer",
    NumberChips = cms.uint32(1),
    NumberL1EGamma = cms.uint32(12),
    NumberL1Jet = cms.uint32(12),
    NumberL1Muon = cms.uint32(8),
    NumberL1Tau = cms.uint32(12),
    NumberPhysTriggers = cms.uint32(512),
    OrderOfChip = cms.vint32(1),
    PinsOnChip = cms.uint32(512)
)
