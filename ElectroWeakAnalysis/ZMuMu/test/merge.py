import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("Merge")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "file:zMuSa-UML-Z.root", "file:zMuSa-UML-W.root"
    )
)

process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('file:./zMuSa-UML.root'),
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
