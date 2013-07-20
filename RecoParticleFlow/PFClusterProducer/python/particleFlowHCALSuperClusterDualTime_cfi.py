import FWCore.ParameterSet.Config as cms

particleFlowHCALSuperClusterDualTime = cms.EDProducer("PFSuperClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # PFCluster collection                                  
    PFClusters = cms.InputTag("particleFlowHCALClusterDualTime"),
    # PFCluster collection HO                                  
    PFClustersHO = cms.InputTag("particleFlowClusterHO")
)

