import FWCore.ParameterSet.Config as cms

hltElePixelSeedsTripletsL1Seeded = cms.EDProducer("FakeStateSeedCreatorFromRegionConsecutiveHitsEDProducer",
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    seedingHitSets = cms.InputTag("hltElePixelHitTripletsL1Seeded")
)
