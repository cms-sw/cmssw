import FWCore.ParameterSet.Config as cms


# Delta-R matching value maps

ak8PFJetsCHSPrunedLinks = cms.EDProducer("RecoJetDeltaRValueMapProducer",
                                         src = cms.InputTag("ak8PFJetsCHS"),
                                         matched = cms.InputTag("ak8PFJetsCHSPruned"),
                                         distMin = cms.double(0.8),
                                         value = cms.string('mass')
                        )

ak8PFJetsCHSTrimmedLinks = cms.EDProducer("RecoJetDeltaRValueMapProducer",
                                         src = cms.InputTag("ak8PFJetsCHS"),
                                         matched = cms.InputTag("ak8PFJetsCHSTrimmed"),                                         
                                         distMin = cms.double(0.8),
                                         value = cms.string('mass')
                        )

ak8PFJetsCHSFilteredLinks = cms.EDProducer("RecoJetDeltaRValueMapProducer",
                                         src = cms.InputTag("ak8PFJetsCHS"),
                                         matched = cms.InputTag("ak8PFJetsCHSFiltered"),                                         
                                         distMin = cms.double(0.8),
                                         value = cms.string('mass')  
                        )
