import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
elasticPlotDQMSource = DQMEDAnalyzer("ElasticPlotDQMSource",
    tagRecHit = cms.untracked.InputTag("totemRPRecHitProducer"),
    tagUVPattern = cms.untracked.InputTag("totemRPUVPatternFinder"),
    tagLocalTrack = cms.untracked.InputTag("totemRPLocalTrackFitter"),
  
    verbosity = cms.untracked.uint32(0),
)
