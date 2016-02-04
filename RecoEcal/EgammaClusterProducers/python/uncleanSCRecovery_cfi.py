import FWCore.ParameterSet.Config as cms


uncleanSCRecovered = cms.EDProducer("UncleanSCRecoveryProducer",

            # input collections:
            cleanBcCollection = cms.InputTag('hybridSuperClusters','hybridBarrelBasicClusters'),
            cleanScCollection = cms.InputTag('hybridSuperClusters',''),
                                    
            uncleanBcCollection = cms.InputTag('hybridSuperClusters','uncleanOnlyHybridBarrelBasicClusters'),
            uncleanScCollection = cms.InputTag('hybridSuperClusters','uncleanOnlyHybridSuperClusters'),
            # names of collections to be produced:
            bcCollection = cms.string('uncleanHybridBarrelBasicClusters'),
            scCollection = cms.string('uncleanHybridSuperClusters'),

            )

                                                           
