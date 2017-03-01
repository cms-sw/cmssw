import FWCore.ParameterSet.Config as cms


# Delta-R matching value maps

ak8PFJetsPuppiSoftDropMass = cms.EDProducer("RecoJetDeltaRValueMapProducer",
                                         src = cms.InputTag("ak8PFJetsPuppi"),
                                         matched = cms.InputTag("ak8PFJetsPuppiSoftDrop"),                                         
                                         distMax = cms.double(0.8),
                                         value = cms.string('mass')  
                        )

