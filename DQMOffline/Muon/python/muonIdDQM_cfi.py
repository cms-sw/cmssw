import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
muonIdDQM = DQMEDAnalyzer('MuonIdDQM',
    inputMuonCollection           = cms.InputTag("muons"),
    inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
    inputCSCSegmentCollection     = cms.InputTag("cscSegments"),
    useTrackerMuons               = cms.untracked.bool(True),
    useGlobalMuons                = cms.untracked.bool(True),
    useTrackerMuonsNotGlobalMuons = cms.untracked.bool(True),
    useGlobalMuonsNotTrackerMuons = cms.untracked.bool(False),
    baseFolder                    = cms.untracked.string("Muons/MuonIdDQM")
)
