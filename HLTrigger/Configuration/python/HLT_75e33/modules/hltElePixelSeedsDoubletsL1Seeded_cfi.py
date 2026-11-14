import FWCore.ParameterSet.Config as cms

hltElePixelSeedsDoubletsL1Seeded = cms.EDProducer("FakeStateSeedCreatorFromRegionConsecutiveHitsEDProducer",
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    seedingHitSets = cms.InputTag("hltElePixelHitDoubletsL1Seeded")
)
