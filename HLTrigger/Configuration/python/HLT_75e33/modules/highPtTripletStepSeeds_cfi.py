import FWCore.ParameterSet.Config as cms

highPtTripletStepSeeds = cms.EDProducer("SeedCreatorFromRegionConsecutiveHitsEDProducer",
    MinOneOverPtError = cms.double(1),
    OriginTransverseErrorMultiplier = cms.double(1),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    SeedMomentumForBOFF = cms.double(5),
    TTRHBuilder = cms.string('WithTrackAngle'),
    forceKinematicWithRegionDirection = cms.bool(False),
    magneticField = cms.string(''),
    mightGet = cms.untracked.vstring(
        'RegionsSeedingHitSets_highPtTripletStepHitTriplets__HLTX'
    ),
    propagator = cms.string('PropagatorWithMaterial'),
    seedingHitSets = cms.InputTag("highPtTripletStepHitTriplets")
)
