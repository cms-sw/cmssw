import FWCore.ParameterSet.Config as cms

l1tEfficiencyMuons_offline = cms.EDAnalyzer("L1TEfficiencyMuons_Offline",

  verbose  = cms.untracked.bool(False),
  
)
