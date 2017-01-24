import FWCore.ParameterSet.Config as cms
                                                      
gsBrokenToGSFixedSuperClustersMap = cms.EDProducer("MapNewToOldSCs",
                                                   oldSC=cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel",processName=cms.InputTag.skipCurrentProcess()),
                                                   newSC=cms.InputTag("particleFlowSuperClusterECALGSFixed","particleFlowSuperClusterECALBarrel"),
                                                   oldRefinedSC=cms.InputTag("particleFlowEGamma",processName=cms.InputTag.skipCurrentProcess()),
                                                   newRefinedSC=cms.InputTag("gsFixedRefinedBarrelSuperClusters")
                                                   )
