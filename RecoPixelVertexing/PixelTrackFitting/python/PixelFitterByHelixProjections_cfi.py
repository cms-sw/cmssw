import FWCore.ParameterSet.Config as cms

PixelFitterByHelixProjections = cms.PSet(
    ComponentName = cms.string('PixelFitterByHelixProjections'),
    TTRHBuilder = cms.string('PixelTTRHBuilderWithoutAngle')
)

