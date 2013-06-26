import FWCore.ParameterSet.Config as cms

process = cms.Process("GenFilterEfficiency")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:gen.root')
)

process.dummy = cms.EDAnalyzer("GenFilterEfficiencyAnalyzer",
                               genFilterInfoTag = cms.InputTag("genFilterEfficiencyProducer")
)

process.p = cms.Path(process.dummy)

