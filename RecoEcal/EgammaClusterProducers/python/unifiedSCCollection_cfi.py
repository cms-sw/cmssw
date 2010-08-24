import FWCore.ParameterSet.Config as cms


nonDuplicatedHybridSuperClusters = cms.EDProducer("UnifiedSCCollectionProducer",
            debugLevel = cms.string('DEBUG'),
            # input collections:
            cleanBcProducer = cms.string('hybridSuperClusters'),
            cleanBcCollection = cms.string('hybridBarrelBasicClusters'),
            cleanScProducer = cms.string('hybridSuperClusters'),
            cleanScCollection = cms.string(''),
            uncleanBcProducer = cms.string('uncleanedHybridSuperClusters'),
            uncleanBcCollection = cms.string('hybridBarrelBasicClusters'),
            uncleanScProducer = cms.string('uncleanedHybridSuperClusters'),
            uncleanScCollection = cms.string(''),
            # names of collections to be produced:
            bcCollection = cms.string('hybridBarrelBasicClusters'),
            scCollection = cms.string('hybridSuperClusters'),
            bcCollectionUncleanOnly = cms.string('uncleanOnlyHybridBarrelBasicClusters'),
            scCollectionUncleanOnly = cms.string('uncleanOnlyHybridSuperClusters'),

            )

                                                           
