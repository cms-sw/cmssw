import FWCore.ParameterSet.Config as cms

hltInitialStepTrajectorySeedsLST = cms.EDProducer('LSTOutputConverter',
    lstOutput = cms.InputTag('hltLST'),
    lstInput = cms.InputTag('hltInputLST'),
    lstPixelSeeds = cms.InputTag('hltInputLST'),
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
        TTRHBuilder = cms.string('hltESPTTRHBuilderWithTrackAngle'),
        forceKinematicWithRegionDirection = cms.bool(False)
    )
)
