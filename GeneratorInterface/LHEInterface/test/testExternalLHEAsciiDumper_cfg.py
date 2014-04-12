import FWCore.ParameterSet.Config as cms

process = cms.Process("dumpLHEAscii")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:lheOutputFile.root')
)

process.load("GeneratorInterface/LHEInterface/ExternalLHEAsciiDumper_cfi")

process.p = cms.Path(process.externalLHEAsciiDumper)


