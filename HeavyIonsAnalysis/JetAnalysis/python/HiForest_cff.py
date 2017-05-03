import FWCore.ParameterSet.Config as cms

HiForest = cms.EDAnalyzer("HiForestInfo",
                          HiForestVersion = cms.string(""),
                          GlobalTagLabel = cms.string(""),
                          inputLines = cms.vstring("",)
)
