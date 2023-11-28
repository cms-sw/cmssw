import FWCore.ParameterSet.Config as cms

hgcalMergeLayerClustersL1Seeded = cms.EDProducer('MergeClusterProducer',
  layerClustersEE = cms.InputTag('hgcalLayerClustersEEL1Seeded'),
  layerClustersHSi = cms.InputTag('hgcalLayerClustersHSiL1Seeded'),
  layerClustersHSci = cms.InputTag('hgcalLayerClustersHSciL1Seeded'),
  time_layerclustersEE = cms.InputTag('hgcalLayerClustersEEL1Seeded', 'timeLayerCluster'),
  time_layerclustersHSi = cms.InputTag('hgcalLayerClustersHSiL1Seeded', 'timeLayerCluster'),
  time_layerclustersHSci = cms.InputTag('hgcalLayerClustersHSciL1Seeded', 'timeLayerCluster'),
  timeClname = cms.string('timeLayerCluster'),
  mightGet = cms.optional.untracked.vstring
)