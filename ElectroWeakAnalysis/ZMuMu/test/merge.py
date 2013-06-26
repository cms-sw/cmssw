import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("Merge")


process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "file:NtupleLooseTestNew_oneshot_all_Test_10_1.root",
    "file:NtupleLooseTestNew_oneshot_all_Test_11_1.root",
    "file:NtupleLooseTestNew_oneshot_all_Test_12_1.root",
    "file:NtupleLooseTestNew_oneshot_all_Test_13_1.root", 
    "file:NtupleLooseTestNew_oneshot_all_Test_14_1.root", 
    )
)

process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('file:Ntuple_ZmmPowheg_36X.root'),
)

process.dummy = cms.EDAnalyzer(
    "EventContentAnalyzer"
    )

process.path = cms.Path(
    process.dummy
    )
  
process.endPath = cms.EndPath( 
    process.eventInfo +
    process.out
)
