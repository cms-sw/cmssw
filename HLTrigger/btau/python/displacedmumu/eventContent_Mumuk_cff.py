import FWCore.ParameterSet.Config as cms

MumukHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltMumukPixelSeedFromL2Candidate_*_*', 
        'keep *_hltCkfTrackCandidatesMumuk_*_*', 
        'keep *_hltCtfWithMaterialTracksMumuk_*_*', 
        'keep *_hltMuTracks_*_*', 
        'keep *_hltMumukAllConeTracks_*_*')
)

