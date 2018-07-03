import FWCore.ParameterSet.Config as cms

process = cms.Process("MTDRecoDump")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:step3.root'
    )
)

process.MTDRecoDump = cms.EDAnalyzer('MTDRecoDump')


process.p = cms.Path(process.MTDRecoDump)
