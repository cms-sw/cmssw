import FWCore.ParameterSet.Config as cms

totemRPDQMSource = cms.EDAnalyzer("TotemRPDQMSource",
    tagStatus = cms.InputTag("totemRPRawToDigi", "RP"),
    tagDigi = cms.InputTag("totemRPRawToDigi", "RP"),
    tagCluster = cms.InputTag("totemRPClusterProducer"),
    tagRecHit = cms.InputTag("totemRPRecHitProducer"),
    tagUVPattern = cms.InputTag("totemRPUVPatternFinder"),
    tagLocalTrack = cms.InputTag("totemRPLocalTrackFitter"),
  
    verbosity = cms.untracked.uint32(0),

    buildCorrelationPlots = cms.untracked.bool(False),
    correlationPlotsLimit = cms.untracked.uint32(50),
    correlationPlotsFilter = cms.untracked.string("default=0,1")
)
