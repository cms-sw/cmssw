import FWCore.ParameterSet.Config as cms

FilterByPartonMaxPt = cms.EDFilter("HerwigMaxPtPartonFilter",
  moduleLabel = cms.untracked.InputTag('generator','unsmeared'),
  MinPt = cms.untracked.double(30.0),
  MaxPt = cms.untracked.double(50.0),
  ProcessID = cms.untracked.int32(0)
)
# foo bar baz
# 3xvyTpf572DaF
# zHsLkvUQTMecn
