import FWCore.ParameterSet.Config as cms

Njettiness = cms.EDProducer("NjettinessAdder",
                            src=cms.InputTag("ak5PFJetsCHS"),
                            cone=cms.double(0.5)
                            )
