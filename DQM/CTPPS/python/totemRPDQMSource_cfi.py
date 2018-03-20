import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
totemRPDQMSource = DQMEDAnalyzer('TotemRPDQMSource',
    tagStatus = cms.InputTag("totemRPRawToDigi", "TrackingStrip"),
    tagDigi = cms.InputTag("totemRPRawToDigi", "TrackingStrip"),
    tagCluster = cms.InputTag("totemRPClusterProducer"),
    tagRecHit = cms.InputTag("totemRPRecHitProducer"),
    tagUVPattern = cms.InputTag("totemRPUVPatternFinder"),
    tagLocalTrack = cms.InputTag("totemRPLocalTrackFitter"),
  
    verbosity = cms.untracked.uint32(0),
)
