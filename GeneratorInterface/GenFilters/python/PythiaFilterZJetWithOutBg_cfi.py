import FWCore.ParameterSet.Config as cms

zj_filter = cms.EDFilter("PythiaFilterZJet",
    MinZPt = cms.untracked.double(18.0),
    MaxZPt = cms.untracked.double(22.0),
    MaxMuonEta = cms.untracked.double(2.5),
    MaxEvents = cms.untracked.int32(10),
    MinMuonPt = cms.untracked.double(3.5)
)


