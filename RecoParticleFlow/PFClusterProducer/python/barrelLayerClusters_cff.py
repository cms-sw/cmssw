import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.barrelLayerClusters_cfi import barrelLayerClusters as barrelLayerClusters_

barrelLayerClustersEB = barrelLayerClusters_.clone(
  recHits = 'particleFlowRecHitECAL',
  plugin = dict(
    outlierDeltaFactor = cms.double(2.7 * 0.0175),
    kappa = cms.double(1),
    maxLayerIndex = cms.int32(0),
    deltac = cms.double(1.8 * 0.0175),
    fractionCutoff = cms.double(0.0),
    doSharing = cms.bool(False),
    type = cms.string('EBCLUE')
  )
)

barrelLayerClustersHB = barrelLayerClusters_.clone(
  recHits = 'particleFlowRecHitHBHE',
  plugin = dict(
    outlierDeltaFactor = cms.double(5 * 0.087),
    kappa = cms.double(0),
    maxLayerIndex = cms.int32(4),
    deltac = cms.double(3 * 0.087),
    fractionCutoff = cms.double(0.0),
    doSharing = cms.bool(False),
    type = cms.string('HBCLUE')
  )
)
