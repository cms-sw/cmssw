import FWCore.ParameterSet.Config as cms

DisplacedMumuHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltMumuPixelSeedFromL2Candidate_*_*', 
        'keep *_hltCkfTrackCandidatesMumu_*_*', 
        'keep *_hltCtfWithMaterialTracksMumu_*_*', 
        'keep *_hltMuTracks_*_*')
)

