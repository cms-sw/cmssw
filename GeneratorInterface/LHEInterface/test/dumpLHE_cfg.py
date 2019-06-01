import FWCore.ParameterSet.Config as cms

process = cms.Process("dumpLHE")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:lhe.root')
)

process.dummy = cms.EDAnalyzer("DummyLHEAnalyzer",
    moduleLabel = cms.untracked.InputTag("externalLHEProducer"),
    dumpEvent = cms.untracked.bool(True),
    dumpHeader = cms.untracked.bool(True)
)

process.p = cms.Path(process.dummy)


