import FWCore.ParameterSet.Config as cms


uncleanSCRecovered = cms.EDProducer("UncleanSCRecoveryProducer",
            debugLevel = cms.string('DEBUG'),
            # input collections:
            cleanBcCollection = cms.InputTag('nonDuplicatedHybridSuperClusters','hybridBarrelBasicClusters'),
            cleanScCollection = cms.InputTag('nonDuplicatedHybridSuperClusters','hybridSuperClusters'),
                                    
            uncleanBcCollection = cms.InputTag('nonDuplicatedHybridSuperClusters','uncleanOnlyHybridBarrelBasicClusters'),
            uncleanScCollection = cms.InputTag('nonDuplicatedHybridSuperClusters','uncleanOnlyHybridSuperClusters'),
            # names of collections to be produced:
            bcCollection = cms.string('uncleanHybridBarrelBasicClusters'),
            scCollection = cms.string('uncleanHybridSuperClusters'),

            )

                                                           
