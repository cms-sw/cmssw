import FWCore.ParameterSet.Config as cms


particleTowerProducer = cms.EDProducer('ParticleTowerProducer',
                                          src    = cms.InputTag('particleFlow')
                                          )

