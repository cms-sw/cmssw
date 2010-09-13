import FWCore.ParameterSet.Config as cms


nonDuplicatedHybridSuperClusters = cms.EDProducer("UnifiedSCCollectionProducer",
            debugLevel = cms.string('NONE'),
            # input collections:
            cleanBcCollection = cms.InputTag('hybridSuperClusters','hybridBarrelBasicClusters'),
            cleanScCollection = cms.InputTag('hybridSuperClusters',''),
            uncleanBcCollection = cms.InputTag('uncleanedHybridSuperClusters','hybridBarrelBasicClusters'),
            uncleanScCollection = cms.InputTag('uncleanedHybridSuperClusters',''),
            # names of collections to be produced:
            bcCollection = cms.string('hybridBarrelBasicClusters'),
            scCollection = cms.string('hybridSuperClusters'),
            bcCollectionUncleanOnly = cms.string('uncleanOnlyHybridBarrelBasicClusters'),
            scCollectionUncleanOnly = cms.string('uncleanOnlyHybridSuperClusters'),

            )

                                                           
