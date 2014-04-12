import FWCore.ParameterSet.Config as cms

siStripClustersToDigis = cms.EDProducer("SiStripClusterToDigiProducer",
                                      ClusterProducer = cms.InputTag('siStripClusters','')
                                      )
