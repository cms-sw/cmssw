import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import (
    ClusterShapeHitFilterESProducer as _ClusterShapeHitFilterESProducer,
)

hltPhase2ClusterShapeHitFilterESProducer = _ClusterShapeHitFilterESProducer.clone(
    PixelShapeFile="RecoPixelVertexing/PixelLowPtUtilities/data/ITShapePhase2_all.par",
    PixelShapeFileL1="RecoPixelVertexing/PixelLowPtUtilities/data/ITShapePhase2_all.par",
)
