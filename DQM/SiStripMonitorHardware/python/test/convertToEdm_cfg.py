import FWCore.ParameterSet.Config as cms

process = cms.Process("ConvertToEDM")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

source = cms.Source("NewEventStreamFileReader",
  fileNames = cms.untracked.vstring(
    'file:/data1/nick/data/USC.00078742.0001.A.storageManager.0.0000.dat'
  ),
  skipEvents = cms.untracked.uint32(0)
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.dump = cms.EDAnalyzer("EventContentAnalyzer")

#process.p = cms.Path( process.dump )

process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("edmOutput.root")
)

process.e = cms.EndPath( process.output )
