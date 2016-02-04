import FWCore.ParameterSet.Config as cms
pfElectronInterestingEcalDetIdEB = cms.EDProducer("InterestingDetIdCollectionProducer",
                                                  basicClustersLabel = cms.InputTag("pfElectronTranslator","pf"),
                                                  recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                                  etaSize = cms.int32(5),
                                                  interestingDetIdCollection = cms.string(''),
                                                  phiSize = cms.int32(5)
                                                  )

pfElectronInterestingEcalDetIdEE = cms.EDProducer("InterestingDetIdCollectionProducer",
                                                  basicClustersLabel = cms.InputTag("pfElectronTranslator","pf"),
                                                  recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                                  etaSize = cms.int32(5),
                                                  interestingDetIdCollection = cms.string(''),
                                                  phiSize = cms.int32(5)
                                                  )
