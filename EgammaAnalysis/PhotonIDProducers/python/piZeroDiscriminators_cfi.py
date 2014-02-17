import FWCore.ParameterSet.Config as cms

# $Id: piZeroDiscriminators_cfi.py,v 1.4 2009/05/26 10:50:33 akyriaki Exp $
# Preshower cluster producer
piZeroDiscriminators = cms.EDProducer("PiZeroDiscriminatorProducer",
    # building preshower clusters
    preshClusterShapeProducer = cms.string('preshowerClusterShape'),
    corrPhoProducer = cms.string('photons'),
    correctedPhotonCollection = cms.string(''),
    preshStripEnergyCut = cms.double(0.0),
    w0 = cms.double(4.2),
    EScorr = cms.int32(1),
    Pi0Association = cms.string('PhotonPi0DiscriminatorAssociationMap'),
    preshPi0Nstrip = cms.int32(5),
    preshClusterShapeCollectionX = cms.string('preshowerXClustersShape'),    
    preshClusterShapeCollectionY = cms.string('preshowerYClustersShape'),
    barrelRecHitCollection = cms.InputTag('reducedEcalRecHitsEB'),
    endcapRecHitCollection = cms.InputTag('reducedEcalRecHitsEE'),
    # DEBUG: very verbose  INFO: minimal printout
    debugLevel = cms.string('INFO')
    #debugLevel = cms.string('DEBUG')
)


