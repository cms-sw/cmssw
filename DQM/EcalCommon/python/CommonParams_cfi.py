import FWCore.ParameterSet.Config as cms

ecalCommonParams = cms.untracked.PSet(
    hltTaskMode = cms.untracked.int32(2) #0 -> Do not produce FED plots; 1 -> Only produce FED plots; 2 -> Do both
)
