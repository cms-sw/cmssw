import FWCore.ParameterSet.Config as cms

l25TauSelectedJets =cms.EDProducer("PFJetToCaloProducer",
                                   Source = cms.InputTag("hltL25TauPFTau")
                                      )
