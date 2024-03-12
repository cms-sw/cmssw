import FWCore.ParameterSet.Config as cms

hltElePixelSeedsTripletsL1Seeded = cms.EDProducer("SeedCreatorFromRegionConsecutiveHitsEDProducer",
    MinOneOverPtError = cms.double(1.0),
    OriginTransverseErrorMultiplier = cms.double(1.0),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    SeedMomentumForBOFF = cms.double(5.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    forceKinematicWithRegionDirection = cms.bool(False),
    magneticField = cms.string('ParabolicMf'),
    propagator = cms.string('PropagatorWithMaterialParabolicMf'),
    seedingHitSets = cms.InputTag("hltElePixelHitTripletsL1Seeded")
)
# foo bar baz
# J7i7n9kZI8Jqf
# XAzs2AToY4UNv
