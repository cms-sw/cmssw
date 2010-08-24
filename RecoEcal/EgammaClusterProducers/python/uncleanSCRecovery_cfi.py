import FWCore.ParameterSet.Config as cms


uncleanSCRecovered = cms.EDProducer("UncleanSCRecoveryProducer",
            debugLevel = cms.string('DEBUG'),
            # input collections:
            cleanBcProducer = cms.string('nonDuplicatedHybridSuperClusters'),
            cleanBcCollection = cms.string('hybridBarrelBasicClusters'),
            cleanScProducer = cms.string('nonDuplicatedHybridSuperClusters'),
            cleanScCollection = cms.string('hybridSuperClusters'),
                                    
            uncleanBcProducer = cms.string('nonDuplicatedHybridSuperClusters'),
            uncleanBcCollection = cms.string('uncleanOnlyHybridBarrelBasicClusters'),
            uncleanScProducer = cms.string('nonDuplicatedHybridSuperClusters'),
            uncleanScCollection = cms.string('uncleanOnlyHybridSuperClusters'),
            # names of collections to be produced:
            bcCollection = cms.string('uncleanHybridBarrelBasicClusters'),
            scCollection = cms.string('uncleanHybridSuperClusters'),

            )

                                                           
