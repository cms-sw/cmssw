import FWCore.ParameterSet.Config as cms

PhotonGainSwitchFlagProducer = cms.EDProducer('PhotonGainSwitchFlagProducer',
                                              src = cms.InputTag('reducedEgamma','reducedGedPhotons'),
                                              ebRecHits = cms.InputTag('ecalMultiAndGSGlobalRecHitEB'))

ElectronGainSwitchFlagProducer = cms.EDProducer('ElectronGainSwitchFlagProducer',
                                                src = cms.InputTag('reducedEgamma', 'reducedGedGsfElectrons'),
                                                ebRecHits = cms.InputTag('ecalMultiAndGSGlobalRecHitEB'))
