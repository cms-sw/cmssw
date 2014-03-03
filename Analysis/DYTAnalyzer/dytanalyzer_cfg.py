import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'file:/afs/cern.ch/work/d/dpagano/private/Ideal_BVTT_3000_0_F_F.root'
    ))

process.demo = cms.EDAnalyzer('DYTAnalyzer')
process.TFileService = cms.Service("TFileService", fileName = cms.string('pt3000.root'))

process.p = cms.Path(process.demo)
