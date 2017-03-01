import FWCore.ParameterSet.Config as cms

siStripClusters = cms.EDProducer("ClusterMCsplitStrips",
  UnsplitClusterProducer = cms.InputTag('siStripClustersUnsplit'),
  ClusterRefiner = cms.PSet(
    moduleTypes = cms.vstring(
      'IB1', 'IB2', 'OB1', 'OB2', 'W1A', 'W2A', 'W3A', 'W1B', 'W2B', 'W3B', 'W4', 'W5', 'W6', 'W7'
      # 'IB1', 'IB2', 'OB1', 'OB2'  # barrel modules only
    )
  )
)
