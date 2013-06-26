import FWCore.ParameterSet.Config as cms

shallowClusters = cms.EDProducer("ShallowClustersProducer",
                                 Clusters=cms.InputTag("siStripClusters"),
                                 Prefix=cms.string("cluster"),
                                 )

