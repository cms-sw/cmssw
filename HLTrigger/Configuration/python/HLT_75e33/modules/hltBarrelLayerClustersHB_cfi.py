import FWCore.ParameterSet.Config as cms

hltBarrelLayerClustersHB = cms.EDProducer('BarrelLayerClusterProducer',
  recHits = cms.InputTag("hltParticleFlowRecHitHBHE"),
  plugin = cms.PSet(
    outlierDeltaFactor = cms.double(2.7 * 0.0175),
    kappa = cms.double(3.5),
    maxLayerIndex = cms.int32(4),
    deltac = cms.double(1.8 * 0.0175),
    fractionCutoff = cms.double(0.0),
    doSharing = cms.bool(False),
    type = cms.string('HBCLUE')
  )
)
