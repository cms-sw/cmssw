import FWCore.ParameterSet.Config as cms

#  BasicCluster producer
cosmicBasicClusters = cms.EDProducer("CosmicClusterProducer",
    VerbosityLevel = cms.string('ERROR'),
    barrelHitProducer = cms.string('ecalRecHit'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),                                 
    endcapHitProducer = cms.string('ecalRecHit'),
    endcapHitCollection = cms.string('EcalRecHitsEE'),                                 
    barrelClusterCollection = cms.string('CosmicBarrelBasicClusters'),
    endcapClusterCollection = cms.string('CosmicEndcapBasicClusters'),
    BarrelSeedThr = cms.double(0.04073),
    BarrelSingleThr = cms.double(0.12221),
    BarrelSecondThr = cms.double(0.04073),                                
    EndcapSeedThr = cms.double(0.044),
    EndcapSingleThr = cms.double(0.135),                                 
    EndcapSecondThr = cms.double(0.044),   
    posCalc_logweight = cms.bool(True),
    posCalc_t0_endcPresh = cms.double(1.2),    
    posCalc_t0_endc = cms.double(3.1),
    posCalc_x0 = cms.double(0.89),
    posCalc_w0 = cms.double(4.2), 
    posCalc_t0_barl = cms.double(7.4),
    endcapShapeAssociation = cms.string('CosmicEndcapShapeAssoc'),                                
    clustershapecollectionEE = cms.string('CosmicEndcapShape'),
    barrelShapeAssociation = cms.string('CosmicBarrelShapeAssoc'),                                
    clustershapecollectionEB = cms.string('CosmicBarrelShape'),
    maskedChannels = cms.untracked.vint32(-1)  
)


