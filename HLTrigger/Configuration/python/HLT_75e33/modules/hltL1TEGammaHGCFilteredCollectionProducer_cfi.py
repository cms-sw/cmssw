import FWCore.ParameterSet.Config as cms

hltL1TEGammaHGCFilteredCollectionProducer = cms.EDProducer("L1TEGammaFilteredCollectionProducer",
    applyQual = cms.bool(True),
    inputTag = cms.InputTag("l1EGammaEEProducer","L1EGammaCollectionBXVWithCuts"),
    maxBX = cms.int32(1),
    minBX = cms.int32(-1),
    minPt = cms.double(5.0),
    qualIsMask = cms.bool(False),
    quality = cms.int32(5),
    scalings = cms.vdouble(3.17445, 1.13219, 0.0)
)
