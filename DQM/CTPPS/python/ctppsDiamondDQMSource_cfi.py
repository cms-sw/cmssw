import FWCore.ParameterSet.Config as cms

ctppsDiamondDQMSource = cms.EDAnalyzer("CTPPSDiamondDQMSource",
    tagStatus = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDigi = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagFEDInfo = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDiamondRecHits = cms.InputTag("ctppsDiamondRecHits"),
    tagDiamondLocalTracks = cms.InputTag("ctppsDiamondLocalTracks"),
    tagLocalTrack = cms.InputTag("totemRPLocalTrackFitter"),
    
    excludeMultipleHits = cms.bool(True),
    minimumStripAngleForTomography = cms.double(1e-3),
  
    verbosity = cms.untracked.uint32(10),
)
