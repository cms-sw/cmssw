import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
totemRPDQMSource = DQMEDAnalyzer('TotemRPDQMSource',
    tagStatus = cms.untracked.InputTag("totemRPRawToDigi", "TrackingStrip"),
    tagDigi = cms.untracked.InputTag("totemRPRawToDigi", "TrackingStrip"),
    tagCluster = cms.untracked.InputTag("totemRPClusterProducer"),
    tagRecHit = cms.untracked.InputTag("totemRPRecHitProducer"),
    tagUVPattern = cms.untracked.InputTag("totemRPUVPatternFinder"),
    tagLocalTrack = cms.untracked.InputTag("totemRPLocalTrackFitter"),
  
    verbosity = cms.untracked.uint32(0),
)
