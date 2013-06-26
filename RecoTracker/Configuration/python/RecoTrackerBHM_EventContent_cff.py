# The following comments couldn't be translated into the new config version:

#Tracks

#Tracks

#Tracks without extra and hits

import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfWithMaterialTracksBeamHaloMuon_*_*', 
        'keep recoTrackExtras_ctfWithMaterialTracksBeamHaloMuon_*_*', 
        'keep TrackingRecHitsOwned_ctfWithMaterialTracksBeamHaloMuon_*_*')
)
#RECO content
RecoTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfWithMaterialTracksBeamHaloMuon_*_*', 
        'keep recoTrackExtras_ctfWithMaterialTracksBeamHaloMuon_*_*', 
        'keep TrackingRecHitsOwned_ctfWithMaterialTracksBeamHaloMuon_*_*')
)
#AOD content
RecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfWithMaterialTracksBeamHaloMuon_*_*')
)

