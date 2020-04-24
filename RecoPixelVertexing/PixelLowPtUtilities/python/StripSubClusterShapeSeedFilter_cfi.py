import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeFilter_cfi import StripSubClusterShapeFilterParams
StripSubClusterShapeSeedFilter = cms.PSet(
    StripSubClusterShapeFilterParams,
    ComponentName = cms.string('StripSubClusterShapeSeedFilter'),
    FilterAtHelixStage = cms.bool(False),
    label = cms.untracked.string("Seeds"),
)

