import FWCore.ParameterSet.Config as cms

process = cms.Process("FTLDump")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:step3.root'
    )
)

process.FTLDump = cms.EDAnalyzer('FTLRecoDump')


process.p = cms.Path(process.FTLDump)
