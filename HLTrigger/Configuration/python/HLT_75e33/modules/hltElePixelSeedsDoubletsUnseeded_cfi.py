import FWCore.ParameterSet.Config as cms

hltElePixelSeedsDoubletsUnseeded = cms.EDProducer("FakeStateSeedCreatorFromRegionConsecutiveHitsEDProducer",
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    seedingHitSets = cms.InputTag("hltElePixelHitDoubletsUnseeded")
)
