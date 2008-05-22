import FWCore.ParameterSet.Config as cms

# Preshower cluster producer
multi5x5SuperClustersWithPreshower = cms.EDProducer("PreshowerClusterProducer",
    preshCalibGamma = cms.double(0.024), ## 0.020

    preshStripEnergyCut = cms.double(0.0),
    preshClusterCollectionY = cms.string('preshowerYClusters'),
    # building endcap association
    assocSClusterCollection = cms.string(''),
    etThresh = cms.double(0.0),
    # building preshower clusters
    preshRecHitProducer = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    preshClusterCollectionX = cms.string('preshowerXClusters'),
    preshCalibPlaneY = cms.double(0.7),
    preshCalibPlaneX = cms.double(1.0),
    preshCalibMIP = cms.double(9e-05), ## 78.5e-6

    endcapSClusterProducer = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapSuperClusters"),
    preshNclust = cms.int32(4),
    debugLevel = cms.string(''), ## switch to 'INFO' to get an extensive print-out

    preshClusterEnergyCut = cms.double(0.0),
    preshSeededNstrip = cms.int32(15)
)


