import FWCore.ParameterSet.Config as cms

HiForestInfo = cms.EDAnalyzer(
    "HiForestInfo",
    HiForestVersion = cms.string(""),
    GlobalTagLabel = cms.string(""),
    info = cms.vstring(""),
)
