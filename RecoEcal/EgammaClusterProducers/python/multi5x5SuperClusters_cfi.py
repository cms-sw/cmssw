import FWCore.ParameterSet.Config as cms

#
# $Id: multi5x5SuperClusters.cfi,v 1.3 2008/05/26 15:10:23 dlevans Exp $
#
# Multi5x5 SuperCluster producer
multi5x5SuperClusters = cms.EDProducer("Multi5x5SuperClusterProducer",
    barrelSuperclusterCollection = cms.string('multi5x5BarrelSuperClusters'),
    endcapEtaSearchRoad = cms.double(0.14),
    barrelClusterCollection = cms.string('multi5x5BarrelBasicClusters'),
    dynamicPhiRoad = cms.bool(False),
    endcapClusterProducer = cms.string('multi5x5BasicClusters'),
    barrelPhiSearchRoad = cms.double(0.8),
    endcapPhiSearchRoad = cms.double(0.6),
    VerbosityLevel = cms.string('ERROR'),
    seedTransverseEnergyThreshold = cms.double(1.0),
    doBarrel = cms.bool(False),
    endcapSuperclusterCollection = cms.string('multi5x5EndcapSuperClusters'),
    barrelEtaSearchRoad = cms.double(0.06),
    # for brem recovery
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
    ),
    doEndcaps = cms.bool(True),
    endcapClusterCollection = cms.string('multi5x5EndcapBasicClusters'),
    barrelClusterProducer = cms.string('multi5x5BasicClusters')
)


