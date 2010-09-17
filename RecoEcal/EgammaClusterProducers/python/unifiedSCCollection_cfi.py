import FWCore.ParameterSet.Config as cms

# This will take cleaned and uncleaned collections and save only cleaned
# and the differences between cleaned and uncleaned
#

hybridSuperClusters = cms.EDProducer("UnifiedSCCollectionProducer",
            # input collections:
            cleanBcCollection   = cms.InputTag('cleanedHybridSuperClusters',
                                               'hybridBarrelBasicClusters'),
            cleanScCollection   = cms.InputTag('cleanedHybridSuperClusters',''),
            uncleanBcCollection = cms.InputTag('uncleanedHybridSuperClusters',
                                               'hybridBarrelBasicClusters'),
            uncleanScCollection = cms.InputTag('uncleanedHybridSuperClusters',''),
            # names of collections to be produced:
            bcCollection = cms.string('hybridBarrelBasicClusters'),
            scCollection = cms.string(''),
            bcCollectionUncleanOnly = cms.string('uncleanOnlyHybridBarrelBasicClusters'),
            scCollectionUncleanOnly = cms.string('uncleanOnlyHybridSuperClusters'),

            )

                                                           
