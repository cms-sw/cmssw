import FWCore.ParameterSet.Config as cms

particleFlowHCALSuperClusterDualTime = cms.EDProducer("PFHCALSuperClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    # PFCluster collection                                  
    PFClusters = cms.InputTag("particleFlowClusterHCAL"),
    # PFCluster collection HO                                  
    PFClustersHO = cms.InputTag("particleFlowClusterHO")
)

