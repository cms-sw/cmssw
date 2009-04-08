import FWCore.ParameterSet.Config as cms

process = cms.Process("LumiFileWriter")
  
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptySource")

process.lfw = cms.EDAnalyzer("LumiFileWriter",                              
                             SourcePort   = cms.untracked.uint32(51002),
                             AquireMode   = cms.untracked.uint32(1),
                             MergedOutDir = cms.untracked.string("."),
                             LumiFileDir  = cms.untracked.string("."),
                             WBMOutDir    = cms.untracked.string("."),
                             NBins = cms.untracked.uint32(425),
                             XMin  = cms.untracked.double(100.0),
                             XMax  = cms.untracked.double(3500.0),
                             MergeFiles    = cms.untracked.bool(False),
                             CreateWebPage = cms.untracked.bool(False),
                             TransferToDBS = cms.untracked.bool(False)
                             )

process.p = cms.Path( process.lfw )

