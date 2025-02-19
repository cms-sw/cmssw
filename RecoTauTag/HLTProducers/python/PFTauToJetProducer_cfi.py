import FWCore.ParameterSet.Config as cms

l25TauSelectedJets =cms.EDProducer("PFTauToJetProducer",
                                   Source = cms.InputTag("hltL25TauPFTau")
                                      )
