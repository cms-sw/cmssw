import FWCore.ParameterSet.Config as cms

# Ecal Preshower rechit producer
ecalPreshowerRecHit = cms.EDProducer("ESRecHitProducer",
                                     ESrechitCollection = cms.string('EcalRecHitsES'),
                                     ESdigiCollection = cms.InputTag("ecalPreshowerDigis"),
                                     algo = cms.string("ESRecHitWorker"),
                                     ESGain = cms.int32(1),
                                     ESBaseline = cms.int32(1000),
                                     ESMIPADC = cms.double(50),
                                     ESMIPkeV = cms.double(81.08)
)
