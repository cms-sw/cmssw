import FWCore.ParameterSet.Config as cms

hltBarrelLayerClustersEBMuonSeeded = cms.EDProducer('BarrelLayerClusterProducer',
  recHits = cms.InputTag("hltParticleFlowRecHitECALMuonSeeded"),
  plugin = cms.PSet(
    outlierDeltaFactor = cms.double(2.7 * 0.0175),
    kappa = cms.double(1),
    maxLayerIndex = cms.int32(0),
    deltac = cms.double(1.8 * 0.0175),
    fractionCutoff = cms.double(0.0),
    doSharing = cms.bool(False),
    type = cms.string('EBCLUE')
  )
)
