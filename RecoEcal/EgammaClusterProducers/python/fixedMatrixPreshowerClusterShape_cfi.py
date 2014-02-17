import FWCore.ParameterSet.Config as cms

# $Id: fixedMatrixPreshowerClusterShape_cfi.py,v 1.2 2008/04/21 03:24:07 rpw Exp $
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


