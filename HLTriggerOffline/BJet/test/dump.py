import FWCore.ParameterSet.Config as cms

process = cms.Process("Dump")
# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.dumpPSetRegistry = cms.EDFilter("DumpPSetRegistry")

process.dump = cms.Path(process.dumpPSetRegistry)

