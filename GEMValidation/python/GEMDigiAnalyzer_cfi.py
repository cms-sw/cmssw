import FWCore.ParameterSet.Config as cms

GEMDigiAnalyzer = cms.EDAnalyzer("GEMDigiAnalyzer",
    verbose = cms.untracked.int32(5),
    inputTagRPC = cms.untracked.InputTag("simMuonRPCDigis"),
    inputTagGEM = cms.untracked.InputTag("simMuonGEMDigis"),
    simInputLabel = cms.untracked.string("g4SimHits"),
    minPt = cms.untracked.double(5.),
    simTrackMatching = cms.PSet(
        # SimHit matching:
        verboseSimHit = cms.untracked.int32(5),
        simMuOnlyCSC = cms.untracked.bool(True),
        simMuOnlyGEM = cms.untracked.bool(True),
        discardEleHitsCSC = cms.untracked.bool(True),
        discardEleHitsGEM = cms.untracked.bool(True),
        simInputLabel = cms.untracked.string('g4SimHits'),
        # GEM digi matching:
        verboseGEMDigi = cms.untracked.int32(0),
        gemDigiInput = cms.untracked.InputTag("simMuonGEMDigis"),
        gemPadDigiInput = cms.untracked.InputTag("simMuonGEMCSCPadDigis"),
        gemCoPadDigiInput = cms.untracked.InputTag("simMuonGEMCSCPadDigis", "Coincidence"),
        minBXGEM = cms.untracked.int32(-1),
        maxBXGEM = cms.untracked.int32(1),
        matchDeltaStripGEM = cms.untracked.int32(1),
        )
)
