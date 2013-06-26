import FWCore.ParameterSet.Config as cms

KFBasedPixelFitter = cms.PSet(
    ComponentName = cms.string('KFBasedPixelFitter'),
    useBeamSpotConstraint = cms.bool(True),
       beamSpotConstraint = cms.InputTag('offlineBeamSpot'),
    propagator = cms.string('PropagatorWithMaterial'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    TTRHBuilder = cms.string('PixelTTRHBuilderWithoutAngle')
)

