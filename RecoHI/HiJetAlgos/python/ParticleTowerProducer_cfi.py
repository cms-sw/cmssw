import FWCore.ParameterSet.Config as cms


particleTowerProducer = cms.EDProducer('ParticleTowerProducer',
                                       src    = cms.InputTag('particleFlowTmp'),
                                       useHF = cms.untracked.bool(True)
                                       )

