import FWCore.ParameterSet.Config as cms

process = cms.Process("ContentTester")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(4) )

process.source = cms.Source("NewEventStreamFileReader",
                            fileNames = cms.untracked.vstring('file:../db/closed/kab.812191607.0007.A.storageManager.00.0000.dat')
                            )

process.contentAna = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(process.contentAna)
