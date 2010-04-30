import FWCore.ParameterSet.Config as cms

# Ecal Preshower rechit producer
ecalPreshowerRecHitFit = cms.EDProducer("ESRecHitProducer",
                                        ESrechitCollection = cms.string('EcalRecHitsESFit'),
                                        ESdigiCollection = cms.InputTag("ecalPreshowerDigis"),
                                        algo = cms.string("ESRecHitWorker"),
                                        ESRecoAlgo = cms.int32(1)
                                        )
