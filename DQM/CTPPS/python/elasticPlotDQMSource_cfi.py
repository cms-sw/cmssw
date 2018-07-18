import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
elasticPlotDQMSource = DQMEDAnalyzer("ElasticPlotDQMSource",
    tagRecHit = cms.InputTag("totemRPRecHitProducer"),
    tagUVPattern = cms.InputTag("totemRPUVPatternFinder"),
    tagLocalTrack = cms.InputTag("totemRPLocalTrackFitter"),
  
    verbosity = cms.untracked.uint32(0),
)
