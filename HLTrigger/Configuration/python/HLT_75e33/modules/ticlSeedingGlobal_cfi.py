import FWCore.ParameterSet.Config as cms

ticlSeedingGlobal = cms.EDProducer("TICLSeedingRegionProducer",
    mightGet = cms.optional.untracked.vstring,
    seedingPSet = cms.PSet(
        algo_verbosity = cms.int32(0),
        type = cms.string('SeedingRegionGlobal')
    )
)
