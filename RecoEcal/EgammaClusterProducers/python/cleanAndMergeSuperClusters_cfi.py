import FWCore.ParameterSet.Config as cms


uncleanedNonDuplicatedHybridSuperClusters = cms.EDProducer("CleanAndMergeProducer",
            # input collections:
          
            cleanScInputTag   = cms.InputTag('hybridSuperClusters'),
            uncleanScInputTag = cms.InputTag('uncleanedHybridSuperClusters'),

            # names of collections to be produced:
            bcCollection = cms.string('hybridBarrelBasicClusters'),
            scCollection = cms.string('hybridBarrelSuperClusters'),
            refScCollection = cms.string('hybridCleanedCollectionRef'),
            # some extras (taken from hybridSuperClusters_cfi.py )
            ecalhitproducer = cms.string('ecalRecHit'),
            ecalhitcollection = cms.string('EcalRecHitsEB'),
            posCalcParameters = cms.PSet( T0_barl      = cms.double(7.4),
                                          T0_endc      = cms.double(3.1), 
                                          T0_endcPresh = cms.double(1.2),
                                          LogWeighted  = cms.bool(True),
                                          W0           = cms.double(4.2),
                                          X0           = cms.double(0.89)
                                          )
            
            )

                                                           
