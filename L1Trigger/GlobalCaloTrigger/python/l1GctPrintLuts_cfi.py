import FWCore.ParameterSet.Config as cms

l1GctPrintLuts = cms.EDAnalyzer("L1GctPrintLuts",
    jetRanksFilename = cms.untracked.string("gctJetRanksContents.txt"),
    jetCountFilename = cms.untracked.string("gctJetCountContents.txt"),
    hfSumLutFilename = cms.untracked.string("gctHfSumLutContents.txt")
                                )


