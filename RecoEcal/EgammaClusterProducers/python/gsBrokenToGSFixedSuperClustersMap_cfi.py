import FWCore.ParameterSet.Config as cms
                                                      
gsBrokenToGSFixedSuperClustersMap = cms.EDProducer("MapNewToOldSCs",
                                                   oldSC=cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel",processName=cms.InputTag.skipCurrentProcess()),
                                                   newSC=cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
                                                   oldRefinedSC=cms.InputTag("particleFlowEGamma"),
                                                   newRefinedSC=cms.InputTag("gsFixedRefinedBarrelSuperClusters")
                                                   )
