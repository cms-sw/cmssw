import FWCore.ParameterSet.Config as cms


dqmFileReader = cms.EDAnalyzer("DQMFileReader",

   FileNames = cms.untracked.vstring(),
   referenceFileName = cms.untracked.string("")
                             
)
