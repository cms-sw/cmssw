import FWCore.ParameterSet.Config as cms

totemRPDiamondDQMSource = cms.EDAnalyzer("TotemRPDiamondDQMSource",
    tagStatus = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDigi = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagFEDInfo = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDiamondRecHits = cms.InputTag("ctppsDiamondRecHits"),
    tagDiamondLocalTracks = cms.InputTag("ctppsDiamondLocalTracks"),
    tagLocalTrack = cms.InputTag("totemRPLocalTrackFitter"),
    
    excludeMultipleHits = cms.bool(True),
  
    verbosity = cms.untracked.uint32(10),
)
