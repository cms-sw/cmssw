import FWCore.ParameterSet.Config as cms

# $Id: multi5x5ClusterShape_cfi.py,v 1.1 2008/05/22 11:13:25 hegner Exp $
# Preshower cluster producer
multi5x5PreshowerClusterShape = cms.EDProducer("PreshowerClusterShapeProducer",
    preshStripEnergyCut = cms.double(0.0),
    # building preshower clusters
    preshRecHitProducer = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    preshPi0Nstrip = cms.int32(5),
    endcapSClusterProducer = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapSuperClusters"),
    #    string corrPhoProducer = "correctedPhotons"
    #    string correctedPhotonCollection   = ""
    PreshowerClusterShapeCollectionX = cms.string('multi5x5PreshowerXClustersShape'),
    PreshowerClusterShapeCollectionY = cms.string('multi5x5PreshowerYClustersShape'),
    # DEBUG: very verbose  INFO: minimal printout
    debugLevel = cms.string('INFO')
)


