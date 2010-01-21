import FWCore.ParameterSet.Config as cms

# $Id: piZeroDiscriminators.cfi,v 1.6 2007/10/18 13:50:23 akyriaki Exp $
# Preshower cluster producer
piZeroDiscriminators = cms.EDProducer("PiZeroDiscriminatorProducer",
    # building preshower clusters
    preshClusterShapeProducer = cms.string('preshowerClusterShape'),
    corrPhoProducer = cms.string('correctedPhotons'),
    correctedPhotonCollection = cms.string(''),
    preshStripEnergyCut = cms.double(0.0),
    barrelClusterShapeMapProducer = cms.string('hybridSuperClusters'),
    Pi0Association = cms.string('PhotonPi0DiscriminatorAssociationMap'),
    preshPi0Nstrip = cms.int32(5),
    endcapClusterShapeMapProducer = cms.string('islandBasicClusters'),
    barrelClusterShapeMapCollection = cms.string('hybridShapeAssoc'),
    # DEBUG: very verbose  INFO: minimal printout
    # string debugLevel         = "INFO"
    debugLevel = cms.string('DEBUG'),
    preshClusterShapeCollectionY = cms.string('preshowerYClustersShape'),
    endcapClusterShapeMapCollection = cms.string('islandEndcapShapeAssoc'),
    preshClusterShapeCollectionX = cms.string('preshowerXClustersShape')
)


