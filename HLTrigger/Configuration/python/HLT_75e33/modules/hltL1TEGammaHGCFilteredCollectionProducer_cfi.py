import FWCore.ParameterSet.Config as cms

hltL1TEGammaHGCFilteredCollectionProducer = cms.EDProducer("L1TEGammaFilteredCollectionProducer",
    applyQual = cms.bool(True),
    inputTag = cms.InputTag("l1tGTProducer", "CL2Photons"),
    maxBX = cms.int32(1),
    minBX = cms.int32(-1),
    minPt = cms.double(5.0),
    qualIsMask = cms.bool(True),
    quality = cms.int32(0b0100),
)
