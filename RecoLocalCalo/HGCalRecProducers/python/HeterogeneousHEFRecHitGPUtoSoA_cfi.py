import FWCore.ParameterSet.Config as cms

HEFRecHitGPUtoSoAProd = cms.EDProducer('HEFRecHitGPUtoSoA',
                                       HEFRecHitGPUTok = cms.InputTag('HEFRecHitGPUProd'))
