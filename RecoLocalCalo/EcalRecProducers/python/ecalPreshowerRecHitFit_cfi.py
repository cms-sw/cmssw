import FWCore.ParameterSet.Config as cms

# Ecal Preshower rechit producer
ecalPreshowerRecHitFit = cms.EDProducer("ESRecHitProducer",
                                        ESrechitCollection = cms.string('EcalRecHitsESFit'),
                                        ESdigiCollection = cms.InputTag("ecalPreshowerDigis"),
                                        algo = cms.string("ESRecHitWorker"),
                                        ESRecoAlgo = cms.untracked.int32(1),
                                        ESGain = cms.int32(2),
                                        ESBaseline = cms.int32(0),
                                        ESMIPADC = cms.double(55),
                                        ESMIPkeV = cms.double(81.08)
                                        )
