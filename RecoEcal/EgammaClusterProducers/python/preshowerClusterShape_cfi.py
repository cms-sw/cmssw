import FWCore.ParameterSet.Config as cms

# Preshower cluster producer
preshowerClusterShape = cms.EDProducer("PreshowerClusterShapeProducer",
    preshStripEnergyCut = cms.double(0.0),
    # building preshower clusters
    preshRecHitProducer = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    preshPi0Nstrip = cms.int32(5),
    #endcapSClusterProducer = cms.InputTag("islandSuperClusters","islandEndcapSuperClusters"),
    endcapSClusterProducer = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapSuperClusters"),
    PreshowerClusterShapeCollectionX = cms.string('preshowerXClustersShape'),
    PreshowerClusterShapeCollectionY = cms.string('preshowerYClustersShape'),
    # DEBUG: very verbose  INFO: minimal printout
    debugLevel = cms.string('INFO')
    #debugLevel = cms.string('DEBUG')
)


