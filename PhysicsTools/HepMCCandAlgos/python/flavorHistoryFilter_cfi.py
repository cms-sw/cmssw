import FWCore.ParameterSet.Config as cms

wbb_me_flavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                        jets = cms.InputTag("iterativeCone5GenJets"),
                                        type = cms.int32(3), # only look at matrix element products
                                        minPt = cms.double(20.0),
                                        minDR = cms.double(0.5),
                                        scheme = cms.string("deltaR"),
                                        verbose = cms.bool(False)
                                        )

wcc_me_flavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        jets = cms.InputTag("iterativeCone5GenJets"),
                                        type = cms.int32(3), # only look at matrix element products
                                        minPt = cms.double(20.0),
                                        minDR = cms.double(0.5),
                                        scheme = cms.string("deltaR"),
                                        verbose = cms.bool(False)
                                        )

wc_fe_flavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        jets = cms.InputTag("iterativeCone5GenJets"),
                                        type = cms.int32(2), # only look at flavor excitation products
                                        minPt = cms.double(20.0),
                                        minDR = cms.double(0.5),
                                        scheme = cms.string("deltaR"),
                                        verbose = cms.bool(False)
                                        )
