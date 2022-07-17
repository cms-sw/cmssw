import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonTracks = cms.EDProducer("HLTMuonTrackSelector",
    copyExtras = cms.untracked.bool(True),
    copyMVA = cms.bool(False),
    copyTrajectories = cms.untracked.bool(False),
    muon = cms.InputTag("hltPhase2L3Muons"),
    originalMVAVals = cms.InputTag("none"),
    track = cms.InputTag("hltPhase2L3MuonMerged")
)
