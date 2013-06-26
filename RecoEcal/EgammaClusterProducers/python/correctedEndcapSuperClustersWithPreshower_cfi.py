import FWCore.ParameterSet.Config as cms

# Preshower cluster producer
correctedEndcapSuperClustersWithPreshower = cms.EDProducer("PreshowerClusterProducer",

    preshStripEnergyCut = cms.double(0.0),
    preshClusterCollectionY = cms.string('preshowerYClusters'),
    # building endcap association
    # name for output association collection
    assocSClusterCollection = cms.string(''),
    etThresh = cms.double(0.0),
    # building preshower clusters
    # input collections
    preshRecHitProducer = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    # name for output collections
    preshClusterCollectionX = cms.string('preshowerXClusters'),

    #InputTag endcapSClusterProducer  = islandSuperClusters:islandEndcapSuperClusters
    endcapSClusterProducer = cms.InputTag("correctedIslandEndcapSuperClusters"),
    preshNclust = cms.int32(4),
    debugLevel = cms.string(''), ## switch to 'INFO' to get an extensive print-out

    preshClusterEnergyCut = cms.double(0.0),
    preshSeededNstrip = cms.int32(15)
)


