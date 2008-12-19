import FWCore.ParameterSet.Config as cms


dr0 = 0.0
dr1 = 0.5
dr2 = 9999.0

# w+bb
# matrix element
# widely separated
wbbMEFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(5),
                                        noutput = cms.int32(2),
                                        flavorSource = cms.vint32(2,3),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(dr1),
                                        maxDR = cms.double(dr2),
                                        verbose = cms.bool(False)
                                        )

# w+bb
# matrix element
# colinear (going to junk pile)
wbbMEComplimentFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(5),
                                        noutput = cms.int32(2),
                                        flavorSource = cms.vint32(2,3),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(dr0),
                                        maxDR = cms.double(dr1),
                                        verbose = cms.bool(False)
                                        )

# w+bb
# gluon splitting
# colinear
wbbGSFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(5),
                                        noutput = cms.int32(2),
                                        flavorSource = cms.vint32(1),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(dr0),
                                        maxDR = cms.double(dr1),
                                        verbose = cms.bool(False)
                                        )

# w+bb
# gluon splitting
# widely separated (going to junk pile)
wbbGSComplimentFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(5),
                                        noutput = cms.int32(2),
                                        flavorSource = cms.vint32(1),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(dr1),
                                        maxDR = cms.double(dr2),
                                        verbose = cms.bool(False)
                                        )


# w+b
# flavor excitation
wbFEFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("bFlavorHistoryProducer", "bPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(5),
                                        noutput = cms.int32(1),
                                        flavorSource = cms.vint32(2,3),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(dr0),
                                        maxDR = cms.double(dr0),
                                        verbose = cms.bool(False)
                                        )

# w+cc
# matrix element
# widely separated
wccMEFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(4),
                                        noutput = cms.int32(2),
                                        flavorSource = cms.vint32(2,3),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(dr1),
                                        maxDR = cms.double(dr2),
                                        verbose = cms.bool(False)
                                        )

# w+cc
# matrix element
# colinear (going to junk pile)
wccMEComplimentFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(4),
                                        noutput = cms.int32(2),
                                        flavorSource = cms.vint32(2,3),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(dr0),
                                        maxDR = cms.double(dr1),
                                        verbose = cms.bool(False)
                                        )


# w+cc
# gluon splitting
# colinear
wccGSFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(4),
                                        noutput = cms.int32(2),
                                        flavorSource = cms.vint32(1),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(dr0),
                                        maxDR = cms.double(dr1),
                                        verbose = cms.bool(False)
                                        )

# w+cc
# gluon splitting
# widely separated (going to junk pile)
wccGSComplimentFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(4),
                                        noutput = cms.int32(2),
                                        flavorSource = cms.vint32(1),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(dr1),
                                        maxDR = cms.double(999.0),
                                        verbose = cms.bool(False)
                                        )
# w+c
# flavor excitation
wcFEFlavorHistoryFilter = cms.EDFilter("FlavorHistoryFilter",
                                        src = cms.InputTag("cFlavorHistoryProducer", "cPartonFlavorHistory"),
                                        scheme = cms.string("deltaR"),
                                        flavor = cms.int32(4),
                                        noutput = cms.int32(1),
                                        flavorSource = cms.vint32(2,3),
                                        minPt = cms.double(-1.0),
                                        minDR = cms.double(dr0),
                                        maxDR = cms.double(dr0),
                                        verbose = cms.bool(False)
                                        )
