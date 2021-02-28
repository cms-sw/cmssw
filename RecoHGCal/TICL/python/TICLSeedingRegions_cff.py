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

ticlSeedingTrkHFNose = _ticlSeedingRegionProducer.clone(
    seedingPSet = _ticlSeedingRegionProducer.seedingPSet.clone(type="SeedingRegionByTracks"),
#    type = dict (
#        detector = "HFNose",
#        cutTk = cms.string('3. < abs(eta) < 4. && pt > 1. && quality("highPurity") && hitPattern().numberOfLostHits("MISSING_OUTER_HITS") < 5'),
#    )
)

ticlSeedingByHFHFNose = _ticlSeedingRegionProducer.clone(
  seedingPSet = _ticlSeedingRegionProducer.seedingPSet.clone(type="SeedingRegionByHF")
)
