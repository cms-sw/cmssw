import FWCore.ParameterSet.Config as cms

HEBRecHitGPUtoSoAProd = cms.EDProducer('HEBRecHitGPUtoSoA',
                                       HEBRecHitGPUTok = cms.InputTag('HEBRecHitGPUProd'))
