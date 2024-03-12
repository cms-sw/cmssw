import FWCore.ParameterSet.Config as cms

l1GctPrintLuts = cms.EDAnalyzer("L1GctPrintLuts",
    jetRanksFilename = cms.untracked.string("gctJetRanksContents.txt"),
    hfSumLutFilename = cms.untracked.string("gctHfSumLutContents.txt")
                                )


# foo bar baz
# i1PQa7SIZp5u9
# kVX43swlIlh67
