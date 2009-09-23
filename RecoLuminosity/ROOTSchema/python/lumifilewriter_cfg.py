import FWCore.ParameterSet.Config as cms

process = cms.Process("LumiROOTFileWriter")
  
process.maxEvents = cms.untracked.Pset(
    input = cms.untracked.int32(-1)
    )

process.source = cms.Source("EmptySource")

process.lfw = cms.EDAnalyzer("LumiFileWriter", 
                             
                             SourcePort   = cms.untracked.uint32(51002)
                             MergedOutDir = cms.untracked.string("/LumiROOTFiles")
                             LumiFileDir  = cms.untracked.string("/LumiROOTFiles")
                             WBMOutDir    = cms.untracked.string("/opt/LumiHTML")
                             
                             # 425 bins = 8 BX per bins
                             NBins = cms.untracked.uint32(425)
                             XMin  = cms.untracked.double(100.0)
                             XMax  = cms.untracked.double(3500.0)
                             
                             MergeFiles    = cms.untracked.bool(false)
                             CreateWebPage = cms.untracked.bool(false)
                             TransferToDBS = cms.untracked.bool(false)
                             )

process.p = cms.Path(lfw)

