import FWCore.ParameterSet.Config as cms

hltBarrelLayerClustersHB = cms.EDProducer('BarrelLayerClusterProducer',
  recHits = cms.InputTag("hltParticleFlowRecHitHBHE"),
  plugin = cms.PSet(
    outlierDeltaFactor = cms.double(5 * 0.087),
    kappa = cms.double(0),
    maxLayerIndex = cms.int32(4),
    deltac = cms.double(3 * 0.087),
    fractionCutoff = cms.double(0.0),
    doSharing = cms.bool(False),
    type = cms.string('HBCLUE')
  )
)
