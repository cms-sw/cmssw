import FWCore.ParameterSet.Config as cms

shallowTrackClusters = cms.EDProducer("ShallowTrackClustersProducer",
                                      Tracks=cms.InputTag("generalTracks",""),
                                      Clusters=cms.InputTag("siStripClusters"),
                                      Prefix=cms.string("tsos"),
                                      Suffix=cms.string(""))
