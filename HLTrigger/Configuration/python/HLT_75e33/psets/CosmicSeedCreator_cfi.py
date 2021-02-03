import FWCore.ParameterSet.Config as cms

CosmicSeedCreator = cms.PSet(
    ComponentName = cms.string('CosmicSeedCreator'),
    MinOneOverPtError = cms.double(1.0),
    OriginTransverseErrorMultiplier = cms.double(1.0),
    SeedMomentumForBOFF = cms.double(5.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    forceKinematicWithRegionDirection = cms.bool(False),
    magneticField = cms.string('ParabolicMf'),
    maxseeds = cms.int32(10000),
    propagator = cms.string('PropagatorWithMaterialParabolicMf')
)