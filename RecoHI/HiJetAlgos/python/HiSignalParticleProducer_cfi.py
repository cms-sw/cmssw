import FWCore.ParameterSet.Config as cms


hiSignalGenParticles = cms.EDProducer('HiSignalParticleProducer',
                                    src    = cms.InputTag('genParticles')
                                    )

