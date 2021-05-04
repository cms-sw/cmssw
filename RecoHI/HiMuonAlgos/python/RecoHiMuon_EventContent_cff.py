import FWCore.ParameterSet.Config as cms

#AOD content
RecoHiMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )
# for re-muons
reRecoMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_remuons_*_*',
        'keep *_*_remuons_*',
        # Tracks known by the Muon obj
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoTrackExtras_standAloneMuons_*_*', 
        'keep TrackingRecHitsOwned_standAloneMuons_*_*', 
        'keep recoTracks_reglobalMuons_*_*', 
        'keep recoTrackExtras_reglobalMuons_*_*', 
        'keep recoTracks_retevMuons_*_*', 
        'keep recoTrackExtras_retevMuons_*_*', 
        'keep recoTracksToOnerecoTracksAssociation_retevMuons_*_*')
)
RecoHiMuonAOD.outputCommands.extend(reRecoMuonAOD.outputCommands)

#RECO content
RecoHiMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )

reRecoMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
	'keep *_MuonSeed_*_*',
        'keep *_ancientMuonSeed_*_*',
        'keep *_mergedStandAloneMuonSeeds_*_*',
        'keep TrackingRecHitsOwned_reglobalMuons_*_*', 
        'keep TrackingRecHitsOwned_retevMuons_*_*',
        'keep recoCaloMuons_recalomuons_*_*')
)
reRecoMuonRECO.outputCommands.extend(reRecoMuonAOD.outputCommands)
RecoHiMuonRECO.outputCommands.extend(reRecoMuonRECO.outputCommands)

#Full Event content 
RecoHiMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )

reRecoMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
reRecoMuonFEVT.outputCommands.extend(reRecoMuonRECO.outputCommands)
RecoHiMuonFEVT.outputCommands.extend(reRecoMuonFEVT.outputCommands)
