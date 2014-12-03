import FWCore.ParameterSet.Config as cms


# Delta-R matching value maps

ak8PFJetsCHSPrunedMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
                                         src = cms.InputTag("ak8PFJetsCHS"),
                                         matched = cms.InputTag("ak8PFJetsCHSPruned"),
                                         distMax = cms.double(0.8),
                                         value = cms.string('mass')
                        )

ak8PFJetsCHSTrimmedMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
                                         src = cms.InputTag("ak8PFJetsCHS"),
                                         matched = cms.InputTag("ak8PFJetsCHSTrimmed"),                                         
                                         distMax = cms.double(0.8),
                                         value = cms.string('mass')
                        )

ak8PFJetsCHSFilteredMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
                                         src = cms.InputTag("ak8PFJetsCHS"),
                                         matched = cms.InputTag("ak8PFJetsCHSFiltered"),                                         
                                         distMax = cms.double(0.8),
                                         value = cms.string('mass')  
                        )


ak8PFJetsCHSSoftDropMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
                                         src = cms.InputTag("ak8PFJetsCHS"),
                                         matched = cms.InputTag("ak8PFJetsCHSSoftDrop"),                                         
                                         distMax = cms.double(0.8),
                                         value = cms.string('mass')  
                        )
