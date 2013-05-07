import FWCore.ParameterSet.Config as cms

GEMRecHitAnalyzer = cms.EDAnalyzer("GEMRecHitAnalyzer",
   verbose = cms.untracked.int32(0),
##    simInputLabel = cms.untracked.string("g4SimHits")
)
