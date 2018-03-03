import FWCore.ParameterSet.Config as cms

elasticPlotDQMSource = cms.EDAnalyzer("ElasticPlotDQMSource",
    tagRecHit = cms.InputTag("totemRPRecHitProducer"),
    tagUVPattern = cms.InputTag("totemRPUVPatternFinder"),
    tagLocalTrack = cms.InputTag("totemRPLocalTrackFitter"),
  
    verbosity = cms.untracked.uint32(0),
)
