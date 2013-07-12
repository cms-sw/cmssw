import FWCore.ParameterSet.Config as cms

# $Id: fixedMatrixPreshowerClusterShape.cfi,v 1.2 2008/03/27 18:05:29 dlevans Exp $
# Preshower cluster producer
fixedMatrixPreshowerClusterShape = cms.EDProducer("PreshowerClusterShapeProducer",
    preshStripEnergyCut = cms.double(0.0),
    # building fixedMatrixPreshower clusters
    preshRecHitProducer = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    preshPi0Nstrip = cms.int32(5),
    endcapSClusterProducer = cms.InputTag("fixedMatrixSuperClusters","fixedMatrixEndcapSuperClusters"),
    #    string corrPhoProducer = "correctedPhotons"
    #    string correctedPhotonCollection   = ""
    PreshowerClusterShapeCollectionX = cms.string('fixedMatrixPreshowerXClustersShape'),
    PreshowerClusterShapeCollectionY = cms.string('fixedMatrixPreshowerYClustersShape'),
    # DEBUG: very verbose  INFO: minimal printout
    debugLevel = cms.string('INFO')
)


