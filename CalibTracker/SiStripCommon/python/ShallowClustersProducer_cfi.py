import FWCore.ParameterSet.Config as cms

shallowClusters = cms.EDProducer("ShallowClustersProducer",
                                 Clusters=cms.InputTag("siStripClusters"),
                                 Prefix=cms.string("cluster"),
                                 )

# foo bar baz
# ovr5702Yf42jU
# C3q8LlQxCr4vQ
