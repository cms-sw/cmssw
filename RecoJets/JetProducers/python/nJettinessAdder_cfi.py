import FWCore.ParameterSet.Config as cms

Njettiness = cms.EDProducer("NjettinessAdder",
                            src=cms.InputTag("ca8PFJetsCHS"),
                            cone=cms.double(0.8),
                            Njets=cms.vuint32(1,2,3)
                            )
