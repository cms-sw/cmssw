import FWCore.ParameterSet.Config as cms

electronIDValueMapProducer = cms.EDProducer('ElectronIDValueMapProducer',
                                          ebReducedRecHitCollection = cms.InputTag('reducedEcalRecHitsEB'),
                                          eeReducedRecHitCollection = cms.InputTag('reducedEcalRecHitsEE'),
                                          esReducedRecHitCollection = cms.InputTag('reducedEcalRecHitsES'),
                                          src = cms.InputTag('gedGsfElectrons'),
                                          dataFormat = cms.string('RECO')
)
