import FWCore.ParameterSet.Config as cms

KFBasedPixelFitter = cms.PSet(
    ComponentName = cms.string('KFBasedPixelFitter'),
    propagator = cms.string('PropagatorWithMaterial'),
    TTRHBuilder = cms.string('PixelTTRHBuilderWithoutAngle')
)

