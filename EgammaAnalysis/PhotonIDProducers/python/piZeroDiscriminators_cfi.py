import FWCore.ParameterSet.Config as cms

# $Id: piZeroDiscriminators_cfi.py,v 1.2 2008/04/21 01:42:22 rpw Exp $
# Preshower cluster producer
piZeroDiscriminators = cms.EDProducer("PiZeroDiscriminatorProducer",
    # building preshower clusters
    preshClusterShapeProducer = cms.string('preshowerClusterShape'),
    corrPhoProducer = cms.string('correctedPhotons'),
    correctedPhotonCollection = cms.string(''),
    preshStripEnergyCut = cms.double(0.0),
    Pi0Association = cms.string('PhotonPi0DiscriminatorAssociationMap'),
    preshPi0Nstrip = cms.int32(5),
    # DEBUG: very verbose  INFO: minimal printout
    debugLevel = cms.string('INFO'),
    preshClusterShapeCollectionY = cms.string('preshowerYClustersShape'),
    barrelRecHitCollection = cms.InputTag('reducedRecHitCollectionEB'),
    endcapRecHitCollection = cms.InputTag('reducedRecHitCollectionEE')
)


