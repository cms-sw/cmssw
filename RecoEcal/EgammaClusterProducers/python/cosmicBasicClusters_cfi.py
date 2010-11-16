import FWCore.ParameterSet.Config as cms

#  BasicCluster producer
cosmicBasicClusters = cms.EDProducer("CosmicClusterProducer",
    endcapHitProducer = cms.string('ecalRecHit'),
    endcapUHitProducer = cms.string('ecalFixedAlphaBetaFitUncalibRecHit'),
    barrelClusterCollection = cms.string('CosmicBarrelBasicClusters'),
    EndcapSecondThr = cms.double(9.99),
    VerbosityLevel = cms.string('ERROR'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    barrelUHitCollection = cms.string('EcalUncalibRecHitsEB'),
    BarrelSingleThr = cms.double(14.99),
    BarrelSupThr = cms.double(2.0),
    EndcapSupThr = cms.double(3.0),
    barrelShapeAssociation = cms.string('CosmicBarrelShapeAssoc'),
    clustershapecollectionEE = cms.string('CosmicEndcapShape'),
    clustershapecollectionEB = cms.string('CosmicBarrelShape'),
    EndcapSingleThr = cms.double(25.99),
    endcapClusterCollection = cms.string('CosmicEndcapBasicClusters'),
    BarrelSecondThr = cms.double(4.99),
    EndcapSeedThr = cms.double(9.99),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    endcapUHitCollection = cms.string('EcalUncalibRecHitsEB'),
    BarrelSeedThr = cms.double(4.99),
    endcapShapeAssociation = cms.string('CosmicEndcapShapeAssoc'),
    barrelHitProducer = cms.string('ecalRecHit'),
    barrelUHitProducer = cms.string('ecalFixedAlphaBetaFitUncalibRecHit'),
    barrelUnHitProducer = cms.string('ecalFixedAlphaBetaFitUncalibRecHit'),
    endcapUnHitProducer = cms.string('ecalFixedAlphaBetaFitUncalibRecHit'),
    barrelUnHitCollection = cms.string('EcalUncalibRecHitsEB'),
    endcapUnHitCollection = cms.string('EcalUncalibRecHitsEE'),
                                     

)
