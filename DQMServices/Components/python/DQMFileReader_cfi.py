import FWCore.ParameterSet.Config as cms


dqmFileReader = cms.EDFilter("DQMFileReader",

   FileNames = cms.untracked.vstring()
                             
)
