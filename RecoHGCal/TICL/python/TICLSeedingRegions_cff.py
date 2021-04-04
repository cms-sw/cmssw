import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.ticlSeedingRegionProducer_cfi import ticlSeedingRegionProducer as _ticlSeedingRegionProducer

# SEEDING REGION

ticlSeedingGlobal = _ticlSeedingRegionProducer.clone(
  seedingPSet = _ticlSeedingRegionProducer.seedingPSet.clone(type="SeedingRegionGlobal")
)

ticlSeedingTrk = _ticlSeedingRegionProducer.clone(
  seedingPSet = _ticlSeedingRegionProducer.seedingPSet.clone(type="SeedingRegionByTracks")
)

ticlSeedingGlobalHFNose = _ticlSeedingRegionProducer.clone(
  seedingPSet = _ticlSeedingRegionProducer.seedingPSet.clone(type="SeedingRegionGlobal")
)

# fixme: empty seed, need to work on propagator
ticlSeedingTrkHFNose = _ticlSeedingRegionProducer.clone(
    seedingPSet = _ticlSeedingRegionProducer.seedingPSet.clone(
        type="SeedingRegionByTracks",
        cutTk = cms.string('3. < abs(eta) < 4. && pt > 1. && quality("highPurity") && hitPattern().numberOfLostHits("MISSING_OUTER_HITS") < 5'),
        detector = cms.string("HFNose")
    )
)

ticlSeedingByHFHFNose = _ticlSeedingRegionProducer.clone(
  seedingPSet = _ticlSeedingRegionProducer.seedingPSet.clone(type="SeedingRegionByHF")
)
