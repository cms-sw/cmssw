import FWCore.ParameterSet.Config as cms

# Ecal Preshower rechit producer
ecalPreshowerRecHit = cms.EDProducer("ESRecHitProducer",
                                     ESrechitCollection = cms.string('EcalRecHitsES'),
                                     ESdigiCollection = cms.InputTag("ecalPreshowerDigis"),
                                     algo = cms.string("ESRecHitWorker"),
                                     ESRecoAlgo = cms.int32(0)
)
