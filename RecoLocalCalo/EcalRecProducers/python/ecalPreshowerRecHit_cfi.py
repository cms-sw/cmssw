import FWCore.ParameterSet.Config as cms

# Ecal Preshower rechit producer
ecalPreshowerRecHit = cms.EDProducer("ESRecHitProducer",
                                     ESrechitCollection = cms.string('EcalRecHitsES'),
                                     ESdigiCollection = cms.InputTag("ecalPreshowerDigis"),
                                     algo = cms.string("ESRecHitWorker"),
                                     ESGain = cms.int32(2),
                                     ESBaseline = cms.int32(1000),
                                     ESMIPADC = cms.double(55),
                                     ESMIPkeV = cms.double(81.08)
)
