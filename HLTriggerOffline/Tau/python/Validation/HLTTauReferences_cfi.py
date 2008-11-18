import FWCore.ParameterSet.Config as cms

TauMCProducer  = cms.EDProducer("HLTTauMCProducer",
                              GenParticles  = cms.untracked.InputTag("source"),
                              ptMinTau      = cms.untracked.double(10),
                              ptMinMuon     = cms.untracked.double(3),
                              ptMinElectron = cms.untracked.double(5),
                              BosonID       = cms.untracked.vint32(23,25,37),
                              EtaMax        = cms.untracked.double(2.5)
)

HLTTauRef = cms.Sequence(TauMCProducer)








                              
