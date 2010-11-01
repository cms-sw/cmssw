import FWCore.ParameterSet.Config as cms

hiSpikeCleaner = cms.EDProducer("HiSpikeCleaner",
                                recHitProducerBarrel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                recHitProducerEndcap = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                originalSuperClusterProducer = cms.InputTag("correctedIslandBarrelSuperClusters"),
                                outputColl  = cms.string( "" ),
                                etCut          = cms.double(10),
                                doTimingCut    = cms.untracked.bool(True),
                                swissCutThr    = cms.untracked.double(0.95)
                                )




