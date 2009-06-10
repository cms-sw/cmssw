import FWCore.ParameterSet.Config as cms

process = cms.Process("ErrDump")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("ErrorStreamSource",
                            fileNames = cms.untracked.vstring('file:../db/closed/kab.812221605.0001.Error.storageManager.00.0000.dat')
                            #fileNames = cms.untracked.vstring('file:/tmp/test1.err')
                            )

process.contentAna = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(process.contentAna)
