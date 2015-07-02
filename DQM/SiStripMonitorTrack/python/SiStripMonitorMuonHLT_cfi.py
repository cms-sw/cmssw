import FWCore.ParameterSet.Config as cms

sistripMonitorMuonHLT = cms.EDAnalyzer("SiStripMonitorMuonHLT",
    verbose = cms.untracked.bool(False),
    normalize = cms.untracked.bool(True),
    printNormalize = cms.untracked.bool(False),
    monitorName = cms.untracked.string("HLT/HLTMonMuon"),
    prescaleEvt = cms.untracked.int32(-1),
    runOnClusters = cms.untracked.bool(True),
    clusterCollectionTag = cms.untracked.InputTag ("hltSiStripRawToClustersFacility"),
    runOnMuonCandidates = cms.untracked.bool(True),
    l3MuonTag = cms.untracked.InputTag ("hltL3MuonCandidates"),
    runOnTracks = cms.untracked.bool(False),
    trackCollectionTag = cms.untracked.InputTag ("hltL3TkTracksFromL2")
)
