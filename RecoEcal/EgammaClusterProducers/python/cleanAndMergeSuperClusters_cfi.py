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
            posCalc_t0 = cms.double(7.4),
            posCalc_logweight = cms.bool(True),
            posCalc_w0 = cms.double(4.2),
            posCalc_x0 = cms.double(0.89),

            
            )

                                                           
