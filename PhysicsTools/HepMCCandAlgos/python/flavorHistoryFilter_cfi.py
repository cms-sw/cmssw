import FWCore.ParameterSet.Config as cms

wbbMEFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(5),
                                        noutput = cms.int32(2),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(0.5),
                                        maxDR = cms.double(99999.0),
                                        verbose = cms.bool(False)
                                        )

wbbGSFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(5),
                                        noutput = cms.int32(2),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(0.0),
                                        maxDR = cms.double(0.5),
                                        verbose = cms.bool(False)
                                        )



wbFEFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(5),
                                        noutput = cms.int32(1),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(0.0),
                                        maxDR = cms.double(9999.0),
                                        verbose = cms.bool(False)
                                        )


wccMEFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(4),
                                        noutput = cms.int32(2),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(0.5),
                                        maxDR = cms.double(99999.0),
                                        verbose = cms.bool(False)
                                        )

wccGSFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(4),
                                        noutput = cms.int32(2),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(0.0),
                                        maxDR = cms.double(0.5),
                                        verbose = cms.bool(False)
                                        )

wcFEFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(4),
                                        noutput = cms.int32(1),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(0.0),
                                        maxDR = cms.double(9999.0),
                                        verbose = cms.bool(False)
                                        )
