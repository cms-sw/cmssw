import FWCore.ParameterSet.Config as cms
#Tracks without extra and hits

#AOD content
RecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
	'keep recoTracks_ctfWithMaterialTracksBeamHaloMuon_*_*')
)

#RECO content
RecoTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoTrackExtras_ctfWithMaterialTracksBeamHaloMuon_*_*', 
        'keep TrackingRecHitsOwned_ctfWithMaterialTracksBeamHaloMuon_*_*')
)
RecoTrackerRECO.outputCommands.extend(RecoTrackerAOD.outputCommands)

#Full Event content 
RecoTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoTrackerFEVT.outputCommands.extend(RecoTrackerRECO.outputCommands)
