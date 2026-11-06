import FWCore.ParameterSet.Config as cms

process = cms.Process("ETLRecoSoADump")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:step3.root'
    )
)

process.ETLRecoSoADump = cms.EDAnalyzer('ETLRecoSoADump')


process.p = cms.Path(process.ETLRecoSoADump)
