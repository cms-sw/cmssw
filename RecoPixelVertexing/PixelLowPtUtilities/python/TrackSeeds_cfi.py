import FWCore.ParameterSet.Config as cms

import RecoTracker.TkSeedGenerator.SeedGeneratorFromProtoTracksEDProducer_cfi
pixelTrackSeeds = RecoTracker.TkSeedGenerator.SeedGeneratorFromProtoTracksEDProducer_cfi.SeedGeneratorFromProtoTracksEDProducer.clone(
    InputCollection = '',
    TTRHBuilder = 'WithTrackAngle'
    )
#pixelTrackSeeds = cms.EDProducer("SeedProducer",
#    TTRHBuilder = cms.string('WithTrackAngle'),
#    tripletList = cms.vstring()
#)

