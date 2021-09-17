import FWCore.ParameterSet.Config as cms

#
#
# Multi5x5 SuperCluster producer
multi5x5SuperClustersCleaned = cms.EDProducer("Multi5x5SuperClusterProducer",
    barrelSuperclusterCollection = cms.string('multi5x5BarrelSuperClusters'),
    endcapEtaSearchRoad = cms.double(0.14),
    barrelClusterTag = cms.InputTag('multi5x5BasicClustersCleaned',
				    'multi5x5BarrelBasicClusters'),
    dynamicPhiRoad = cms.bool(False),
    endcapClusterTag= cms.InputTag('multi5x5BasicClustersCleaned',
				   'multi5x5EndcapBasicClusters'),
    barrelPhiSearchRoad = cms.double(0.8),
    endcapPhiSearchRoad = cms.double(0.6),
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
  
)


multi5x5SuperClustersUncleaned = multi5x5SuperClustersCleaned.clone(
    barrelClusterTag = 'multi5x5BasicClustersUncleaned:multi5x5BarrelBasicClusters',
    endcapClusterTag = 'multi5x5BasicClustersUncleaned:multi5x5EndcapBasicClusters'
)

multi5x5SuperClusters=cms.EDProducer("UnifiedSCCollectionProducer",
            # input collections:
            cleanBcCollection   = cms.InputTag('multi5x5BasicClustersCleaned',
                                               'multi5x5EndcapBasicClusters'),
            cleanScCollection   = cms.InputTag('multi5x5SuperClustersCleaned',
                                               'multi5x5EndcapSuperClusters'),
            uncleanBcCollection = cms.InputTag('multi5x5BasicClustersUncleaned',
                                               'multi5x5EndcapBasicClusters'),
            uncleanScCollection = cms.InputTag('multi5x5SuperClustersUncleaned',
                                               'multi5x5EndcapSuperClusters'),
            # names of collections to be produced:
            bcCollection = cms.string('multi5x5EndcapBasicClusters'),
            scCollection = cms.string('multi5x5EndcapSuperClusters'),
            bcCollectionUncleanOnly = cms.string('uncleanOnlyMulti5x5EndcapBasicClusters'),
            scCollectionUncleanOnly = cms.string('uncleanOnlyMulti5x5EndcapSuperClusters'),

            )
