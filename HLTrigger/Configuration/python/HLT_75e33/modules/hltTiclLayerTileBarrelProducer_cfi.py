import FWCore.ParameterSet.Config as cms

hltTiclLayerTileBarrelProducer = cms.EDProducer("TICLLayerTileProducer",
  detector = cms.string('Barrel'),
  layer_HFNose_clusters = cms.InputTag("hgcalLayerClustersHFNose"),
  layer_clusters = cms.InputTag("hltMergeLayerClusters"),
)
