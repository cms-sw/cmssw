import FWCore.ParameterSet.Config as cms

particleFlowSuperClusterHCALUpgrade = cms.EDProducer("PFSuperClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # PFCluster collection                                  
    PFClusters = cms.InputTag("particleFlowClusterHCALUpgrade"),
    # PFCluster collection HO                                  
    PFClustersHO = cms.InputTag("particleFlowClusterHO")
)

