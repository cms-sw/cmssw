import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("RecoMET.METAnalyzers.metSequence_cff")

process.load("RecoMET.METAnalyzers.exampleanalyzer_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/home/fgolf/CMSSW_3_1_2/src/FEAFAB96-5683-DE11-A10B-001AA00AA41B.root'
    )
)

process.p = cms.Path(process.metCorSequence*process.demo)
