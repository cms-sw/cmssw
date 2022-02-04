import FWCore.ParameterSet.Config as cms

ticlSeedingTrk = cms.EDProducer("TICLSeedingRegionProducer",
    mightGet = cms.optional.untracked.vstring,
    seedingPSet = cms.PSet(
        algo_verbosity = cms.int32(0),
        type = cms.string('SeedingRegionByTracks')
    )
)
