import FWCore.ParameterSet.Config as cms

interestingDetIdFromSuperClusterProducer = cms.EDProducer("InterestingDetIdFromSuperClusterProducer",
                                                    superClustersLabel = cms.InputTag(''),
                                                    recHitsLabel = cms.InputTag('ecalRecHit','EcalRecHitsEB'),
                                                    etaSize = cms.int32(5),
                                                    phiSize = cms.int32(5),
                                                    # keep hits with severity >= severityLevel
                                                    # set to negative to skip 
                                                    severityLevel = cms.int32(1),
                                                    interestingDetIdCollection = cms.string(''),
                                                    keepNextToDead = cms.bool(True),
                                                    keepNextToBoundary = cms.bool(True)
                                                    )
