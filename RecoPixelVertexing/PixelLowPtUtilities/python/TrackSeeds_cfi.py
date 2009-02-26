import FWCore.ParameterSet.Config as cms

pixelTrackSeeds = cms.EDProducer("SeedProducer",
    TTRHBuilder = cms.string('WithTrackAngle'),
    tripletList = cms.vstring()
)

