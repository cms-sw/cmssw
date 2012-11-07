import FWCore.ParameterSet.Config as cms

particleFlowSuperClusterECAL = cms.EDProducer("PFSuperClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
                                              
    #PFClusters collection
    PFClusters = cms.InputTag("particleFlowClusterECAL"),
    PFClustersES = cms.InputTag("particleFlowClusterPS"),
                                              
    PFBasicClusterCollectionBarrel = cms.string("particleFlowBasicClusterECALBarrel"),                                       
    PFSuperClusterCollectionBarrel = cms.string("particleFlowSuperClusterECALBarrel"),
    PFBasicClusterCollectionEndcap = cms.string("particleFlowBasicClusterECALEndcap"),                                       
    PFSuperClusterCollectionEndcap = cms.string("particleFlowSuperClusterECALEndcap"),
    PFBasicClusterCollectionPreshower = cms.string("particleFlowSuperClusterECALPreshower"),
    PFSuperClusterCollectionEndcapWithPreshower = cms.string("particleFlowSuperClusterECALEndcapWithPreshower"),                                          
    
    # threshold in ECAL
    thresh_PFClusterSeedBarrel = cms.double(3.0),
    thresh_PFClusterBarrel = cms.double(0.5),

    thresh_PFClusterSeedEndcap = cms.double(3.0),
    thresh_PFClusterEndcap = cms.double(0.5),

    # window width in ECAL
    phiwidth_SuperClusterBarrel = cms.double(0.28),
    etawidth_SuperClusterBarrel = cms.double(0.04),

    phiwidth_SuperClusterEndcap = cms.double(0.28),
    etawidth_SuperClusterEndcap = cms.double(0.04),

    # threshold in preshower
    thresh_PFClusterES = cms.double(0.),                                          

    # threshold for clusters outside mustache area
    doMustachePUcleaning = cms.bool(False),                                          
    #thresh_PFClusterMustacheOutBarrel = cms.double(0.),
    #thresh_PFClusterMustacheOutEndcap = cms.double(0.),                                             

    #corrections
    applyCrackCorrections = cms.bool(False)                                          
                                              
)


