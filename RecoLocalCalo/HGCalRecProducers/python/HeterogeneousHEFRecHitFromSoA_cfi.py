import FWCore.ParameterSet.Config as cms

HEFRecHitFromSoAProd = cms.EDProducer('HEFRecHitFromSoA',
                                      HEFRecHitSoATok = cms.InputTag('HEFRecHitGPUtoSoAProd'))
