import FWCore.ParameterSet.Config as cms

l1GctPrintLuts = cms.EDAnalyzer("L1GctPrintLuts",
    filename = cms.untracked.string("gctLutContents.txt")
                                )


