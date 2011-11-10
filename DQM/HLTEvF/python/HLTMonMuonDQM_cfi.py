import FWCore.ParameterSet.Config as cms

hltMonMuDQM = cms.EDAnalyzer("HLTMuonDQMSource",
# Settings for Heavy Ion
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    reqNum = cms.uint32(1),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    l3MuonSeedTag = cms.untracked.InputTag("hltHIL3TrajectorySeed"),
    l3MuonTag = cms.untracked.InputTag("hltHIL3MuonCandidates"),
    l3MuonSeedTagOIState = cms.untracked.InputTag("hltHIL3TrajSeedOIState"),
    l3MuonSeedTagOIHit = cms.untracked.InputTag("hltHIL3TrajSeedOIHit"),
    l3MuonTrkFindingOIState = cms.untracked.InputTag("hltHIL3TrackCandidateFromL2OIState"),
    l3MuonTrkFindingOIHit = cms.untracked.InputTag("hltHIL3TrackCandidateFromL2OIHit"),
    l3MuonTrkOIState = cms.untracked.InputTag("hltHIL3TkTracksFromL2OIState"),
    l3MuonTrkOIHit = cms.untracked.InputTag("hltHIL3TkTracksFromL2OIHit"),
    l3MuonTrk = cms.untracked.InputTag("hltHIL3TkTracksFromL2"),
    l3muons = cms.untracked.InputTag("hltHIL3Muons"),
    l3muonsOIState = cms.untracked.InputTag("hltHIL3MuonsOIState"),
    l3muonsOIHit = cms.untracked.InputTag("hltHIL3MuonsOIHit"),
    TrigResultInput = cms.InputTag('TriggerResults','','HLT'),
    filters = cms.VPSet(
      # HI L1 Muons
      cms.PSet(
        directoryName = cms.string('HIL1PassThrough'),
        triggerBits = cms.vstring('HLT_HIL1DoubleMuOpen_v1', 'HLT_HIL1DoubleMu0_HighQ_v1')
      ),
      # HI L2 Muons
      cms.PSet(
        directoryName = cms.string('HIL2PassThrough'),
        triggerBits = cms.vstring('HLT_HIL2Mu3_NHitQ_v1', 'HLT_HIL2Mu7_v1', 'HLT_HIL2Mu15_v1', 'HLT_HIL2DoubleMu0_v1', 'HLT_HIL2DoubleMu3_v1'),
      ),
      # HI L3 Muons
      cms.PSet(
        directoryName = cms.string('HIL3PassThrough'),
        triggerBits = cms.vstring('HLT_HIL3Mu3_v1', 'HLT_HIL3DoubleMuOpen_v1', 'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v1'),
      ),
      # HI Single Mu
      cms.PSet(
        directoryName = cms.string('HISingleMu'),
        triggerBits = cms.vstring('HLT_HIL2Mu3_NHitQ_v1', 'HLT_HIL2Mu7_v1', 'HLT_HIL2Mu15_v1', 'HLT_HIL3Mu3_v1'),
      ),
      # HI Double Mu
      cms.PSet(
        directoryName = cms.string('HIDoubleMu'),
        triggerBits = cms.vstring('HLT_HIL1DoubleMu0_HighQ_v1', 'HLT_HIL2DoubleMu0_v1', 'HLT_HIL2DoubleMu3_v1', 'HLT_HIL3DoubleMuOpen_v1', 'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v1'),
      ),
    ),
    disableROOToutput = cms.untracked.bool(True)
)

