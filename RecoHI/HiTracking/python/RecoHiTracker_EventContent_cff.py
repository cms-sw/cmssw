import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoHiTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_globalPrimTracks_*_*', 
        'keep recoTrackExtras_globalPrimTracks_*_*', 
        'keep TrackingRecHitsOwned_globalPrimTracks_*_*',
		'keep recoTracks_selectHiTracks_*_*', 
        'keep recoTrackExtras_selectHiTracks_*_*', 
        'keep TrackingRecHitsOwned_selectHiTracks_*_*'
    )
)
#RECO content
RecoHiTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_globalPrimTracks_*_*', 
        'keep recoTrackExtras_globalPrimTracks_*_*', 
        'keep TrackingRecHitsOwned_globalPrimTracks_*_*',
		'keep recoTracks_selectHiTracks_*_*', 
        'keep recoTrackExtras_selectHiTracks_*_*', 
        'keep TrackingRecHitsOwned_selectHiTracks_*_*'
    )
)
#AOD content
RecoHiTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_selectHiTracks_*_*'
    )
)
