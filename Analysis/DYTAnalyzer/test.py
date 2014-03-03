import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    'file:../../myOutputFile.root'
    ))

process.demo = cms.EDAnalyzer('DYTAnalyzer')
process.TFileService = cms.Service("TFileService", fileName = cms.string('output.root'))

process.p = cms.Path(process.demo)
