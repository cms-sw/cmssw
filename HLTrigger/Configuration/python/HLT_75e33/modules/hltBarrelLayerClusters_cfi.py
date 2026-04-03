import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.barrelLayerClusters_cff import barrelLayerClustersEB as _barrelLayerClustersEB
from RecoParticleFlow.PFClusterProducer.barrelLayerClusters_cff import barrelLayerClustersHB as _barrelLayerClustersHB

hltBarrelLayerClustersEB = _barrelLayerClustersEB.clone(
    recHits = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
)

hltBarrelLayerClustersEB = barrelLayerClusters_.clone(
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

hltBarrelLayerClustersEBL1Seeded = cms.EDProducer('BarrelLayerClusterProducer',
  recHits = cms.InputTag("hltParticleFlowRecHitECALL1Seeded"),
  plugin = cms.PSet(
    outlierDeltaFactor = cms.double(2.7 * 0.0175),
    kappa = cms.double(3.5),
    maxLayerIndex = cms.int32(0),
    deltac = cms.double(1.8 * 0.0175),
    fractionCutoff = cms.double(0.0),
    doSharing = cms.bool(False),
    type = cms.string('EBCLUE')
  )
)

hltBarrelLayerClustersHB = _barrelLayerClustersHB.clone(
    recHits = cms.InputTag("hltParticleFlowRecHitHBHE"),
)
