# The following comments couldn't be translated into the new config version:

#Tracks

#Tracks

#Tracks without extra and hits

import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfWithMaterialTracksTIF_*_*', 
        'keep recoTrackExtras_ctfWithMaterialTracksTIF_*_*', 
        'keep TrackingRecHitsOwned_ctfWithMaterialTracksTIF_*_*', 
        'keep recoTracks_rsWithMaterialTracksTIF_*_*', 
        'keep recoTrackExtras_rsWithMaterialTracksTIF_*_*', 
        'keep TrackingRecHitsOwned_rsWithMaterialTracksTIF_*_*', 
        'keep recoTracks_cosmictrackfinderTIF_*_*', 
        'keep recoTrackExtras_cosmictrackfinderTIF_*_*', 
        'keep TrackingRecHitsOwned_cosmictrackfinderTIF_*_*')
)
#RECO content
RecoTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfWithMaterialTracksTIF_*_*', 
        'keep recoTrackExtras_ctfWithMaterialTracksTIF_*_*', 
        'keep TrackingRecHitsOwned_ctfWithMaterialTracksTIF_*_*', 
        'keep recoTracks_rsWithMaterialTracksTIF_*_*', 
        'keep recoTrackExtras_rsWithMaterialTracksTIF_*_*', 
        'keep TrackingRecHitsOwned_rsWithMaterialTracksTIF_*_*', 
        'keep recoTracks_cosmictrackfinderTIF_*_*', 
        'keep recoTrackExtras_cosmictrackfinderTIF_*_*', 
        'keep TrackingRecHitsOwned_cosmictrackfinderTIF_*_*')
)
#AOD content
RecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfWithMaterialTracksTIF_*_*', 
        'keep recoTracks_rsWithMaterialTracksTIF_*_*', 
        'keep recoTracks_cosmictrackfinderTIF_*_*')
)

