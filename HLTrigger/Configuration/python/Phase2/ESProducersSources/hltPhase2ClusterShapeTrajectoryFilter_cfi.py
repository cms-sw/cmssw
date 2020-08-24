import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import (
    ClusterShapeTrajectoryFilter as _ClusterShapeTrajectoryFilter,
)

### This is a PSet, bad idea to change its label.
ClusterShapeTrajectoryFilter = _ClusterShapeTrajectoryFilter.clone(
    cacheSrc = cms.InputTag("hltPhase2siPixelClusterShapeCache")
)
