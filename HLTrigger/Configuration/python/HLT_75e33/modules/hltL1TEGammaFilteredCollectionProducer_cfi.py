import FWCore.ParameterSet.Config as cms

hltL1TEGammaFilteredCollectionProducer = cms.EDProducer("L1TEGammaFilteredCollectionProducer",
    applyQual = cms.bool(True),
    inputTag = cms.InputTag("l1tEGammaClusterEmuProducer"),
    maxBX = cms.int32(1),
    minBX = cms.int32(-1),
    minPt = cms.double(5.0),
    qualIsMask = cms.bool(True),
    quality = cms.int32(2),
    scalings = cms.vdouble(2.6604, 1.06077, 0.0)
)
