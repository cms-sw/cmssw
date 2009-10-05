import FWCore.ParameterSet.Config as cms

sistripMonitorMuonHLT = cms.EDAnalyzer("SiStripMonitorMuonHLT",
    outputFile = cms.untracked.string('./SiStripMuonHLTDQM.root'),
    #disableROOToutput = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False),
    monitorName = cms.untracked.string("HLT/HLTMonMuon"),
    prescaleEvt = cms.untracked.int32(-1),
    clusterCollectionTag = cms.untracked.InputTag ("hltSiStripRawToClustersFacility"),
    l3MuonTag = cms.untracked.InputTag ("hltL3MuonCandidates")
)
