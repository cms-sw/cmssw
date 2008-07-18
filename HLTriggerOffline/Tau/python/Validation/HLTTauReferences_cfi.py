import FWCore.ParameterSet.Config as cms

TauMCProducer  = cms.EDProducer("HLTTauMCProducer",
                              GenParticles  = cms.untracked.InputTag("source"),
                              ptMinTau      = cms.untracked.double(15),
                              ptMinMuon     = cms.untracked.double(14),
                              ptMinElectron = cms.untracked.double(12),
                              BosonID       = cms.untracked.int32(23),
                              EtMax         = cms.untracked.double(2.5)
)

HLTTauRef = cms.Sequence(TauMCProducer)








                              
