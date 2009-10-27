
import FWCore.ParameterSet.Config as cms

process = cms.Process("Convert")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.source = cms.Source("NewEventStreamFileReader",
  fileNames = cms.untracked.vstring("file:DATFILE")
)

maxEvents = cms.PSet(
  input = cms.untracked.int32(-1)
)

process.anal = cms.EDAnalyzer("EventContentAnalyzer")

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("ROOFILE"),
  outputCommands = cms.untracked.vstring("drop *", "keep FEDRawDataCollection_*_*_*")
)

process.p = cms.Path(process.anal)
process.e = cms.EndPath(process.out)
