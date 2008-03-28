import FWCore.ParameterSet.Config as cms

# Preshower cluster producer
correctedEndcapSuperClustersWithPreshower = cms.EDProducer("PreshowerClusterProducer",
    preshCalibGamma = cms.double(0.024), ## 0.020

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
    preshCalibPlaneY = cms.double(0.7),
    preshCalibPlaneX = cms.double(1.0),
    preshCalibMIP = cms.double(9e-05), ## 78.5e-6 

    #InputTag endcapSClusterProducer  = islandSuperClusters:islandEndcapSuperClusters
    endcapSClusterProducer = cms.InputTag("correctedIslandEndcapSuperClusters"),
    preshNclust = cms.int32(4),
    debugLevel = cms.string(''), ## switch to 'INFO' to get an extensive print-out

    preshClusterEnergyCut = cms.double(0.0),
    preshSeededNstrip = cms.int32(15)
)


