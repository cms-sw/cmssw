import FWCore.ParameterSet.Config as cms
import os

particleFlowSuperClusterECALBox = cms.EDProducer(
    "PFECALSuperClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    #clustering type: "Box" or "Mustache"
    ClusteringType = cms.string("Box"),
    #energy weighting: "Raw", "CalibratedNoPS", "CalibratedTotal"
    EnergyWeight = cms.string("Raw"),

    #this overrides both dphi cuts below if true!
    useDynamicDPhiWindow = cms.bool(False),
    
    #PFClusters collection
    PFClusters = cms.InputTag("particleFlowClusterECAL"),
    ESAssociation = cms.InputTag("particleFlowClusterECAL"),
    BeamSpot = cms.InputTag("offlineBeamSpot"),    
                                                  
    PFBasicClusterCollectionBarrel = cms.string("particleFlowBasicClusterECALBarrel"),                                       
    PFSuperClusterCollectionBarrel = cms.string("particleFlowSuperClusterECALBarrel"),
    PFBasicClusterCollectionEndcap = cms.string("particleFlowBasicClusterECALEndcap"),                                       
    PFSuperClusterCollectionEndcap = cms.string("particleFlowSuperClusterECALEndcap"),
    PFBasicClusterCollectionPreshower = cms.string("particleFlowBasicClusterECALPreshower"),
    PFSuperClusterCollectionEndcapWithPreshower = cms.string("particleFlowSuperClusterECALEndcapWithPreshower"),                                          

    #use preshower ?
    use_preshower = cms.bool(True),

    # are the seed thresholds Et or Energy?
    seedThresholdIsET = cms.bool(True),

    # regression setup
    useRegression = cms.bool(False), #regressions are mustache only
    regressionConfig = cms.PSet(
       regressionKeyEB = cms.string('pfscecal_EBCorrection_offline_v2'),
       uncertaintyKeyEB = cms.string('pfscecal_EBUncertainty_offline_v2'),
       regressionKeyEE = cms.string('pfscecal_EECorrection_offline_v2'),
       uncertaintyKeyEE = cms.string('pfscecal_EEUncertainty_offline_v2'),
       vertexCollection = cms.InputTag("offlinePrimaryVertices"),
       ecalRecHitsEB = cms.InputTag('ecalRecHit','EcalRecHitsEB'),
       ecalRecHitsEE = cms.InputTag('ecalRecHit','EcalRecHitsEE')
       ),

    #threshold for final SuperCluster Et
    thresh_SCEt = cms.double(4.0),    
    
    # threshold in ECAL
    thresh_PFClusterSeedBarrel = cms.double(3.0),
    thresh_PFClusterBarrel = cms.double(0.5),

    thresh_PFClusterSeedEndcap = cms.double(5.0),
    thresh_PFClusterEndcap = cms.double(0.5),

    # window width in ECAL
    phiwidth_SuperClusterBarrel = cms.double(0.28),
    etawidth_SuperClusterBarrel = cms.double(0.04),

    phiwidth_SuperClusterEndcap = cms.double(0.28),
    etawidth_SuperClusterEndcap = cms.double(0.04),

    # turn on merging of the seed cluster to its nearest neighbors
    # that share a rechit
    doSatelliteClusterMerge = cms.bool(False),
    satelliteClusterSeedThreshold = cms.double(50.0),
    satelliteMajorityFraction = cms.double(0.5),
    dropUnseedable = cms.bool(False),
    #thresh_PFClusterMustacheOutBarrel = cms.double(0.),
    #thresh_PFClusterMustacheOutEndcap = cms.double(0.),                                             

    #corrections
    applyCrackCorrections = cms.bool(False)                                          
                                              
)

particleFlowSuperClusterECALMustache = cms.EDProducer(
    "PFECALSuperClusterProducer",
    # verbosity 
    verbose = cms.untracked.bool(False),
    #clustering type: "Box" or "Mustache"
    ClusteringType = cms.string("Mustache"),
    #energy weighting: "Raw", "CalibratedNoPS", "CalibratedTotal"
    EnergyWeight = cms.string("Raw"),

    #this overrides both dphi cuts below if true!
    useDynamicDPhiWindow = cms.bool(True), 
                                              
    #PFClusters collection
    PFClusters = cms.InputTag("particleFlowClusterECAL"),
    ESAssociation = cms.InputTag("particleFlowClusterECAL"),
    BeamSpot = cms.InputTag("offlineBeamSpot"),
                                              
    PFBasicClusterCollectionBarrel = cms.string("particleFlowBasicClusterECALBarrel"),                                       
    PFSuperClusterCollectionBarrel = cms.string("particleFlowSuperClusterECALBarrel"),
    PFBasicClusterCollectionEndcap = cms.string("particleFlowBasicClusterECALEndcap"),                                       
    PFSuperClusterCollectionEndcap = cms.string("particleFlowSuperClusterECALEndcap"),
    PFBasicClusterCollectionPreshower = cms.string("particleFlowBasicClusterECALPreshower"),
    PFSuperClusterCollectionEndcapWithPreshower = cms.string("particleFlowSuperClusterECALEndcapWithPreshower"),                                          

    # are the seed thresholds Et or Energy?
    seedThresholdIsET = cms.bool(True),
    # regression setup
    useRegression = cms.bool(True),
    regressionConfig = cms.PSet(
       regressionKeyEB = cms.string('pfscecal_EBCorrection_offline_v2'),
       uncertaintyKeyEB = cms.string('pfscecal_EBUncertainty_offline_v2'),
       regressionKeyEE = cms.string('pfscecal_EECorrection_offline_v2'),
       uncertaintyKeyEE = cms.string('pfscecal_EEUncertainty_offline_v2'),
       vertexCollection = cms.InputTag("offlinePrimaryVertices"),
       ecalRecHitsEB = cms.InputTag('ecalRecHit','EcalRecHitsEB'),
       ecalRecHitsEE = cms.InputTag('ecalRecHit','EcalRecHitsEE')
       ),
       
    #threshold for final SuperCluster Et
    thresh_SCEt = cms.double(4.0),
    
    # threshold in ECAL
    thresh_PFClusterSeedBarrel = cms.double(1.0),
    thresh_PFClusterBarrel = cms.double(0.0),

    thresh_PFClusterSeedEndcap = cms.double(1.0),
    thresh_PFClusterEndcap = cms.double(0.0),

    # window width in ECAL ( these don't mean anything for Mustache )
    phiwidth_SuperClusterBarrel = cms.double(0.6),
    etawidth_SuperClusterBarrel = cms.double(0.04),

    phiwidth_SuperClusterEndcap = cms.double(0.6),
    etawidth_SuperClusterEndcap = cms.double(0.04),

    # threshold in preshower
    thresh_PFClusterES = cms.double(0.),           

    # turn on merging of the seed cluster to its nearest neighbors
    # that share a rechit
    doSatelliteClusterMerge = cms.bool(False),
    satelliteClusterSeedThreshold = cms.double(50.0),
    satelliteMajorityFraction = cms.double(0.5),
    dropUnseedable = cms.bool(False),
    #thresh_PFClusterMustacheOutBarrel = cms.double(0.),
    #thresh_PFClusterMustacheOutEndcap = cms.double(0.), 

    #corrections
    applyCrackCorrections = cms.bool(False)
                                              
)

#define the default clustering type
particleFlowSuperClusterECAL = particleFlowSuperClusterECALMustache.clone()

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(particleFlowSuperClusterECAL, useDynamicDPhiWindow = False)
pp_on_AA_2018.toModify(particleFlowSuperClusterECAL, phiwidth_SuperClusterBarrel = 0.20)
pp_on_AA_2018.toModify(particleFlowSuperClusterECAL, phiwidth_SuperClusterEndcap = 0.20)
