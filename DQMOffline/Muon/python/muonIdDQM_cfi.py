import FWCore.ParameterSet.Config as cms

muonIdDQM = DQMStep1Module('MuonIdDQM',
    inputMuonCollection           = cms.InputTag("muons"),
    inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
    inputCSCSegmentCollection     = cms.InputTag("cscSegments"),
    useTrackerMuons               = cms.untracked.bool(True),
    useGlobalMuons                = cms.untracked.bool(True),
    useTrackerMuonsNotGlobalMuons = cms.untracked.bool(True),
    useGlobalMuonsNotTrackerMuons = cms.untracked.bool(False),
    baseFolder                    = cms.untracked.string("Muons/MuonIdDQM")
)
