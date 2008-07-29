import FWCore.ParameterSet.Config as cms

flavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                   src = cms.InputTag("flavorHistoryProducer", "bpartonFlavorHistory"),
                                   jets = cms.InputTag("sisCone5GenJets"),
                                   type = cms.int32(3), # only look at matrix element products
                                   minPt = cms.double(10.0),
                                   minDR = cms.double(0.5),
                                   scheme = cms.string("deltaR"),
                                   verbose = cms.bool(False)
                                   )
