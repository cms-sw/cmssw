import FWCore.ParameterSet.Config as cms

#
# $Id: cosmicSuperClusters_cfi.py,v 1.4 2009/11/26 17:52:42 dlevans Exp $
#
# Moved to Multi5x5SuperCluster producer
cosmicSuperClusters = cms.EDProducer("Multi5x5SuperClusterProducer",

    VerbosityLevel = cms.string('ERROR'),
    endcapClusterProducer = cms.string('cosmicBasicClusters'),
    barrelClusterProducer = cms.string('cosmicBasicClusters'),
    endcapClusterCollection = cms.string('CosmicEndcapBasicClusters'),
    barrelClusterCollection = cms.string('CosmicBarrelBasicClusters'),
    endcapSuperclusterCollection = cms.string('CosmicEndcapSuperClusters'),                                 
    barrelSuperclusterCollection = cms.string('CosmicBarrelSuperClusters'),
    barrelEtaSearchRoad = cms.double(0.20),
    barrelPhiSearchRoad = cms.double(0.55),                                 
    endcapEtaSearchRoad = cms.double(0.14),
    endcapPhiSearchRoad = cms.double(0.6),
    seedTransverseEnergyThreshold = cms.double(0.0),
    doBarrel = cms.bool(True),
    doEndcaps = cms.bool(True),

    # if dynamicPhiRoad is set to false the parameters below
    # are not used and the standard fixed phi roads are used
    dynamicPhiRoad = cms.bool(False),
    bremRecoveryPset = cms.PSet(
        barrel = cms.PSet(
            cryVec = cms.vint32(16, 13, 11, 10, 9,
                8, 7, 6, 5, 4,
                3),
            cryMin = cms.int32(2),
            etVec = cms.vdouble(5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 45.0, 55.0, 135.0, 195.0,
                225.0)
        ),
        endcap = cms.PSet(
            a = cms.double(47.85),
            c = cms.double(0.1201),
            b = cms.double(108.8)
        )
    )


)

