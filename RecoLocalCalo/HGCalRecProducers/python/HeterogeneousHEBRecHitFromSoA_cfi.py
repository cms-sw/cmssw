import FWCore.ParameterSet.Config as cms

HEBRecHitFromSoAProd = cms.EDProducer('HEBRecHitFromSoA',
                                      HEBRecHitSoATok = cms.InputTag('HEBRecHitGPUtoSoAProd'))
