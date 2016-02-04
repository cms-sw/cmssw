import FWCore.ParameterSet.Config as cms

PixelFitterByConformalMappingAndLine = cms.PSet(
    ComponentName = cms.string('PixelFitterByConformalMappingAndLine'),
    TTRHBuilder = cms.string('PixelTTRHBuilderWithoutAngle')
)

