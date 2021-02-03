import FWCore.ParameterSet.Config as cms

multi5x5PreshowerClusterShape = cms.EDProducer("PreshowerClusterShapeProducer",
    PreshowerClusterShapeCollectionX = cms.string('multi5x5PreshowerXClustersShape'),
    PreshowerClusterShapeCollectionY = cms.string('multi5x5PreshowerYClustersShape'),
    debugLevel = cms.string('INFO'),
    endcapSClusterProducer = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    preshPi0Nstrip = cms.int32(5),
    preshRecHitProducer = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    preshStripEnergyCut = cms.double(0.0)
)
