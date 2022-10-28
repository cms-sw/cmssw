import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTANALYSIS")

process.maxEvents.input = -1

process.source = cms.Source("PoolSource",
    setRunNumber = cms.untracked.uint32(621),
    fileNames = cms.untracked.vstring('file:noparentdict_step1.root')
)

process.analysis = cms.EDAnalyzer("IntTestAnalyzer",
    valueMustMatch = cms.untracked.int32(1),
    moduleLabel = cms.untracked.InputTag("intProduct"),
)

process.p = cms.Path(process.analysis)
