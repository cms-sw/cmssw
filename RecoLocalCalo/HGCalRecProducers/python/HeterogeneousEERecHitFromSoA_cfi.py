import FWCore.ParameterSet.Config as cms

EERecHitFromSoAProd = cms.EDProducer('EERecHitFromSoA',
                                     EERecHitSoATok = cms.InputTag('EERecHitGPUtoSoAProd'))
