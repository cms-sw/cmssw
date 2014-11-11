import FWCore.ParameterSet.Config as cms

electronIDValueMapProducer = cms.EDProducer('ElectronIDValueMapProducer',
                                          ebReducedRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                          eeReducedRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                          esReducedRecHitCollection = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
                                          src = cms.InputTag('gedGsfElectrons'),
                                          dataFormat = cms.string('RECO')
)
