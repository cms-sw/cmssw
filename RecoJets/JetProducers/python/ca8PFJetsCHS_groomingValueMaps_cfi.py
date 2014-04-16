import FWCore.ParameterSet.Config as cms


# Delta-R matching value maps

ca8PFJetsCHSPrunedLinks = cms.EDProducer("RecoJetDeltaRValueMapProducer",
                                         src = cms.InputTag("ca8PFJetsCHS"),
                                         matched = cms.InputTag("ca8PFJetsCHSPruned"),
                                         distMin = cms.double(0.8),
                                         value = cms.string('mass')
                        )

ca8PFJetsCHSTrimmedLinks = cms.EDProducer("RecoJetDeltaRValueMapProducer",
                                         src = cms.InputTag("ca8PFJetsCHS"),
                                         matched = cms.InputTag("ca8PFJetsCHSTrimmed"),                                         
                                         distMin = cms.double(0.8),
                                         value = cms.string('mass')
                        )

ca8PFJetsCHSFilteredLinks = cms.EDProducer("RecoJetDeltaRValueMapProducer",
                                         src = cms.InputTag("ca8PFJetsCHS"),
                                         matched = cms.InputTag("ca8PFJetsCHSFiltered"),                                         
                                         distMin = cms.double(0.8),
                                         value = cms.string('mass')  
                        )
