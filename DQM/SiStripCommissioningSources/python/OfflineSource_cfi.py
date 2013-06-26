import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring()
)
maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
