import FWCore.ParameterSet.Config as cms

from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching

MuonDigiAnalyzer = cms.EDAnalyzer("MuonDigiAnalyzer",
    verbose = cms.untracked.int32(5),
    inputTagRPC = cms.untracked.InputTag("simMuonRPCDigis"),
    inputTagGEM = cms.untracked.InputTag("simMuonGEMDigis"),
    simInputLabel = cms.untracked.string("g4SimHits"),
    minPt = cms.untracked.double(5.),
    simTrackMatching = SimTrackMatching
)
