import FWCore.ParameterSet.Config as cms

from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching

GEMRecHitAnalyzer = cms.EDAnalyzer("GEMRecHitAnalyzer",
   verbose = cms.untracked.int32(5),
    inputTagGEM = cms.untracked.InputTag("simMuonGEMDigis"),
    simInputLabel = cms.untracked.string("g4SimHits"),
    minPt = cms.untracked.double(5.),
    simTrackMatching = SimTrackMatching
)
