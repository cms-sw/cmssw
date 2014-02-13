import FWCore.ParameterSet.Config as cms

process = cms.Process("USER")

## MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

## Options and Output Report
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## Source
process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(
            '/store/relval/CMSSW_7_0_0_pre11/RelValProdTTbar/GEN-SIM-RECO/START70_V4-v1/00000/0EA82C3C-646A-E311-9CB3-0025905A6070.root'
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
