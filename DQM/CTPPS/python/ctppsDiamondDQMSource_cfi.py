import FWCore.ParameterSet.Config as cms

ctppsDiamondDQMSource = cms.EDAnalyzer("CTPPSDiamondDQMSource",
    tagStatus = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDigi = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagFEDInfo = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDiamondRecHits = cms.InputTag("ctppsDiamondRecHits"),
    tagDiamondLocalTracks = cms.InputTag("ctppsDiamondLocalTracks"),
    tagLocalTrack = cms.InputTag("totemRPLocalTrackFitter"),
    
    excludeMultipleHits = cms.bool(True),
    minimumStripAngleForTomography = cms.double(0),
    maximumStripAngleForTomography = cms.double(1),

    offsetsOOT = cms.VPSet( # cut on the OOT bin for physics hits
        # 2016, after TS2
        cms.PSet(
            validityRange = cms.EventRange("1:min - 292520:max"),
            centralOOT = cms.int32(1),
        ),
        # 2017
        cms.PSet(
            validityRange = cms.EventRange("292521:min - 999999999:max"),
            centralOOT = cms.int32(3),
        ),
    ),
  
    verbosity = cms.untracked.uint32(10),
)
