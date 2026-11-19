import FWCore.ParameterSet.Config as cms

hltElePixelSeedsTripletsUnseeded = cms.EDProducer("FakeStateSeedCreatorFromRegionConsecutiveHitsEDProducer",
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    seedingHitSets = cms.InputTag("hltElePixelHitTripletsUnseeded")
)
