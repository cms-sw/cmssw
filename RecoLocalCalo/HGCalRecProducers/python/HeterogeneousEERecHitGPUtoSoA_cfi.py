import FWCore.ParameterSet.Config as cms

EERecHitGPUtoSoAProd = cms.EDProducer('EERecHitGPUtoSoA',
                                      EERecHitGPUTok = cms.InputTag('EERecHitGPUProd'))
