import FWCore.ParameterSet.Config as cms

# Ecal Preshower rechit producer
ecalPreshowerRecHitAnalytic = cms.EDProducer("ESRecHitProducer",
                                        ESrechitCollection = cms.string('EcalRecHitsESFit'),
                                        ESdigiCollection = cms.InputTag("ecalPreshowerDigis"),
                                        algo = cms.string("ESRecHitWorker"),
                                        ESRecoAlgo = cms.int32(2)
                                        )
# foo bar baz
# hFJ4ZV3LLM9jw
# pNKqz3JGBi4P9
