import FWCore.ParameterSet.Config as cms

process = cms.Process("USER")

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

## Options and Output Report
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## Source
process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(
            '/store/relval/CMSSW_3_3_0/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V8-v1/0001/3291E09D-67B7-DE11-9ED6-003048678C9A.root'
                )
                            )
## Maximal Number of Events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('testTFileService.root')
                                   )


process.testAna = cms.EDAnalyzer("TestTFileServiceAnalyzer",
                                 dir1 = cms.string("mydir1"),
                                 dir2 = cms.string("mydir2")
                                 )

process.p = cms.Path( process.testAna )
