import FWCore.ParameterSet.Config as cms

hltEgammaElectronPixelSeedsUnseeded = cms.EDProducer("ElectronSeedFitter",
    MinOneOverPtError = cms.double(1.0),
    OriginTransverseErrorMultiplier = cms.double(1.0),
    SeedMomentumForBOFF = cms.double(5.0),
    TTRHBuilder = cms.string('hltESPTTRHBuilderWithTrackAngle'),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    eleSeedCollection = cms.InputTag("hltEgammaUnfittedElectronPixelSeedsUnseeded"),
    magneticField = cms.string('ParabolicMf'),
    originHalfLength = cms.double(12.5),
    originRadius = cms.double(0.2),
    propagator = cms.string('PropagatorWithMaterialParabolicMf'),
    ptMin = cms.double(1.5)
)