import FWCore.ParameterSet.Config as cms

siStripClustersToDigis = cms.EDProducer("SiStripClusterToDigiProducer",
                                      ClusterProducer = cms.InputTag('siStripClusters','')
                                      )
# foo bar baz
# jeta5n4GVGI7R
