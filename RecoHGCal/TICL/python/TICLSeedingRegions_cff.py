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
