import FWCore.ParameterSet.Config as cms

All                              = cms.uint32(0)           # dummy options - always true
AllGlobalMuons                   = cms.uint32(1)           # checks isGlobalMuon flag
AllStandAloneMuons               = cms.uint32(2)           # checks isStandAloneMuon flag
AllTrackerMuons                  = cms.uint32(3)           # checks isTrackerMuon flag
TrackerMuonArbitrated            = cms.uint32(4)           # resolve ambiguity of sharing segments
AllArbitrated                    = cms.uint32(5)           # all muons with the tracker muon arbitrated
GlobalMuonPromptTight            = cms.uint32(6)           # global muons with tighter fit requirements
TMLastStationLoose               = cms.uint32(7)           # penetration depth loose selector
TMLastStationTight               = cms.uint32(8)           # penetration depth tight selector
TM2DCompatibilityLoose           = cms.uint32(9)           # likelihood based loose selector
TM2DCompatibilityTight           = cms.uint32(10)          # likelihood based tight selector
TMOneStationLoose                = cms.uint32(11)          # require one well matched segment
TMOneStationTight                = cms.uint32(12)          # require one well matched segment
TMLastStationOptimizedLowPtLoose = cms.uint32(13)          # combination of TMLastStation and TMOneStation
TMLastStationOptimizedLowPtTight = cms.uint32(14)          # combination of TMLastStation and TMOneStation
