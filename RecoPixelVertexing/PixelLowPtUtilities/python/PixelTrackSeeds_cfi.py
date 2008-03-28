import FWCore.ParameterSet.Config as cms

pixelTrackSeeds = cms.EDProducer("PixelTrackSeedProducer",
    TTRHBuilder = cms.string('WithTrackAngle'),
    tripletList = cms.vstring()
)


