import FWCore.ParameterSet.Config as cms

uncleanedOnlyMulti5x5SuperClustersWithPreshower = cms.EDProducer("PreshowerPhiClusterProducer",
    assocSClusterCollection = cms.string(''),
    endcapSClusterProducer = cms.InputTag("multi5x5SuperClusters","uncleanOnlyMulti5x5EndcapSuperClusters"),
    esPhiClusterDeltaEta = cms.double(0.15),
    esPhiClusterDeltaPhi = cms.double(0.12),
    esStripEnergyCut = cms.double(0.0),
    etThresh = cms.double(0.0),
    preshClusterCollectionX = cms.string('preshowerXClusters'),
    preshClusterCollectionY = cms.string('preshowerYClusters'),
    preshRecHitProducer = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES")
)
