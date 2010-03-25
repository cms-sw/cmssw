import FWCore.ParameterSet.Config as cms

# Ecal Preshower rechit producer
ecalPreshowerRecHit = cms.EDProducer("ESRecHitProducer",
                                     ESrechitCollection = cms.string('EcalRecHitsES'),
                                     ESdigiCollection = cms.InputTag("ecalPreshowerDigis"),
                                     algo = cms.string("ESRecHitWorker"),
                                     ESRecoAlgo = cms.untracked.int32(0),
                                     ESWeights = cms.untracked.vdouble(0.0, 0.725, 0.4525)
)
