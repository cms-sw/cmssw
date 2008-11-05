import FWCore.ParameterSet.Config as cms

# NOTE:
# Types are:
# 3 = matrix element
# 2 = flavor excitation
# 1 = gluon splitting
# 0 = none
#-1 = any

wbbMEFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                        type = cms.int32(3),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(0.5),
                                        maxDR = cms.double(99999.0),
                                        scheme = cms.string("deltaR"),
       					requireSisters = cms.bool(True),
                                        verbose = cms.bool(False)
                                        )

wbbGSFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                        type = cms.int32(1),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(0.0),
                                        maxDR = cms.double(0.5),
                                        scheme = cms.string("deltaR"),
       					requireSisters = cms.bool(True),
                                        verbose = cms.bool(False)
                                        )

wccMEFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        type = cms.int32(3),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(0.5),
                                        maxDR = cms.double(99999.0),
                                        scheme = cms.string("deltaR"),
       					requireSisters = cms.bool(True),
                                        verbose = cms.bool(False)
                                        )

wccGSFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        type = cms.int32(1),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(0.0),
                                        maxDR = cms.double(0.5),
                                        scheme = cms.string("deltaR"),
       					requireSisters = cms.bool(True),
                                        verbose = cms.bool(False)
                                        )

wcFEFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        type = cms.int32(2),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(0.0),
                                        maxDR = cms.double(99999.0),
                                        scheme = cms.string("deltaR"),
       					requireSisters = cms.bool(False),
                                        verbose = cms.bool(False)
                                        )
