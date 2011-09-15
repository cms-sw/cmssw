import FWCore.ParameterSet.Config as cms

interestingDetIdCollectionProducer = cms.EDProducer("InterestingDetIdCollectionProducer",
                                                    basicClustersLabel = cms.InputTag(''),
                                                    recHitsLabel = cms.InputTag('ecalRecHit','EcalRecHitsEB'),
                                                    etaSize = cms.int32(5),
                                                    phiSize = cms.int32(5),
                                                    severityLevel = cms.int32(1),
                                                    interestingDetIdCollection = cms.string('')
                                                    )
