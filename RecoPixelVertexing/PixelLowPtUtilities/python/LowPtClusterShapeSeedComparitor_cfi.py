import FWCore.ParameterSet.Config as cms

LowPtClusterShapeSeedComparitor = cms.PSet(
    ComponentName = cms.string('LowPtClusterShapeSeedComparitor'),
    clusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache'),
    clusterShapeHitFilter = cms.string('ClusterShapeHitFilter')
)
