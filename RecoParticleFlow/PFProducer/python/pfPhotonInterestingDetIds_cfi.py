import FWCore.ParameterSet.Config as cms
pfPhotonInterestingEcalDetIdEB = cms.EDProducer("InterestingDetIdCollectionProducer",
                                                basicClustersLabel = cms.InputTag("pfPhotonTranslator","pfphot"),
                                                recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                                etaSize = cms.int32(5),
                                                interestingDetIdCollection = cms.string(''),
                                                phiSize = cms.int32(5)
                                                )

pfPhotonInterestingEcalDetIdEE = cms.EDProducer("InterestingDetIdCollectionProducer",
                                                basicClustersLabel = cms.InputTag("pfPhotonTranslator","pfphot"),
                                                recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                                etaSize = cms.int32(5),
                                                interestingDetIdCollection = cms.string(''),
                                                phiSize = cms.int32(5)
                                                )
