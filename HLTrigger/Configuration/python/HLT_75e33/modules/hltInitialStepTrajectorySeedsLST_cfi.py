import FWCore.ParameterSet.Config as cms

hltInitialStepTrajectorySeedsLST = cms.EDProducer('LSTOutputConverter',
    lstOutput = cms.InputTag('hltLST'),
    phase2OTHits = cms.InputTag('hltPhase2OTHitsInputLST'),
    lstPixelSeeds = cms.InputTag('hltPixelSeedInputLST'),
    includeT5s = cms.bool(True),
    includeNonpLSTSs = cms.bool(True),
    propagatorAlong = cms.ESInputTag('', 'PropagatorWithMaterial'),
    propagatorOpposite = cms.ESInputTag('', 'PropagatorWithMaterialOpposite'),
    SeedCreatorPSet = cms.PSet(
        ComponentName = cms.string('SeedFromConsecutiveHitsCreator'),
        propagator = cms.string('PropagatorWithMaterial'),
        SeedMomentumForBOFF = cms.double(5),
        OriginTransverseErrorMultiplier = cms.double(1),
        MinOneOverPtError = cms.double(1),
        magneticField = cms.string(''),
        TTRHBuilder = cms.string('WithTrackAngle'),
        forceKinematicWithRegionDirection = cms.bool(False)
    )
)
