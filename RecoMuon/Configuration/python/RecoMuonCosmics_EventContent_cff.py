import FWCore.ParameterSet.Config as cms

#Add Isolation
from RecoMuon.MuonIsolationProducers.muIsolation_EventContent_cff import *
# AOD content
RecoMuonAOD = cms.PSet(
    
    outputCommands = cms.untracked.vstring(
    # inclusive cosmic reco
    'keep recoTracks_cosmicMuons_*_*', 
    'keep recoTrackExtras_cosmicMuons_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuons_*_*', 
    'keep recoTracks_globalCosmicMuons_*_*', 
    'keep recoTrackExtras_globalCosmicMuons_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicMuons_*_*', 
    'keep recoTracks_tevMuons_*_*',
    'keep recoTrackExtras_tevMuons_*_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_*_*',
    'keep recoMuons_muons_*_*',
    'keep recoMuonTimeExtraedmValueMap_muons_*_*',

    # Split tracks
    'keep recoTracks_globalCosmicSplitMuons_*_*', 
    'keep recoTrackExtras_globalCosmicSplitMuons_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicSplitMuons_*_*', 
    'keep recoMuons_splitMuons_*_*', 
    'keep recoMuonTimeExtraedmValueMap_splitMuons_*_*',
    

    # cosmic reco without RPC
    'keep recoTracks_cosmicMuonsNoRPC_*_*', 
    'keep recoTrackExtras_cosmicMuonsNoRPC_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuonsNoRPC_*_*', 
    'keep recoTracks_globalCosmicMuonsNoRPC_*_*', 
    'keep recoTrackExtras_globalCosmicMuonsNoRPC_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicMuonsNoRPC_*_*', 
    'keep recoMuons_muonsNoRPC_*_*', 
  

    # cosimic reco "1 Leg type" in barrel only
    'keep recoTracks_cosmicMuons1Leg_*_*', 
    'keep recoTrackExtras_cosmicMuons1Leg_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuons1Leg_*_*', 
    'keep recoTracks_globalCosmicMuons1Leg_*_*', 
    'keep recoTrackExtras_globalCosmicMuons1Leg_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicMuons1Leg_*_*', 
    'keep recoMuons_muons1Leg_*_*', 
    'keep recoMuonTimeExtraedmValueMap_muons1Leg_*_*',
    
    # cosimic reco with t0 correction in DTs
    'keep recoTracks_cosmicMuonsWitht0Correction_*_*', 
    'keep recoTrackExtras_cosmicMuonsWitht0Correction_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuonsWitht0Correction_*_*', 
    'keep recoTracks_globalCosmicMuonsWitht0Correction_*_*', 
    'keep recoTrackExtras_globalCosmicMuonsWitht0Correction_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicMuonsWitht0Correction_*_*', 
    'keep recoMuons_muonsWitht0Correction_*_*', 
    'keep recoMuonTimeExtraedmValueMap_muonsWitht0Correction_*_*',

    # cosimic reco in endcaps only  for Beam halo reco
    'keep recoTracks_cosmicMuonsEndCapsOnly_*_*', 
    'keep recoTrackExtras_cosmicMuonsEndCapsOnly_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuonsEndCapsOnly_*_*', 
    'keep recoTracks_globalBeamHaloMuonEndCapslOnly_*_*', 
    'keep recoTrackExtras_globalBeamHaloMuonEndCapslOnly_*_*', 
    'keep TrackingRecHitsOwned_globalBeamHaloMuonEndCapslOnly_*_*',
    'keep recoMuons_muonsBeamHaloEndCapsOnly_*_*',
    'keep recoMuonTimeExtraedmValueMap_muonsBeamHaloEndCapsOnly_*_*',
    
    # lhc like reco
    'keep recoTracks_standAloneMuons_*_*', 
    'keep recoTrackExtras_standAloneMuons_*_*',
    'keep TrackingRecHitsOwned_standAloneMuons_*_*', 
    'keep recoMuons_lhcSTAMuons_*_*',
    'keep recoMuonTimeExtraedmValueMap_lhcSTAMuons_*_*',

    # Tracker Collections
    'keep recoTracks_ctfWithMaterialTracksP5_*_*', 
    'keep recoTracks_ctfWithMaterialTracksBeamHaloMuon_*_*',
    'keep recoTracks_ctfWithMaterialTracksP5LHCNavigation_*_*')
    )
# RECO content
RecoMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_CosmicMuonSeed_*_*', 
                                           'keep *_CosmicMuonSeedEndCapsOnly_*_*', 
                                           'keep *_CosmicMuonSeedWitht0Correction_*_*', 
                                           'keep *_ancientMuonSeed_*_*') 
)
# Full Event content 
RecoMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoMuonRECO.outputCommands.extend(RecoMuonAOD.outputCommands)
RecoMuonFEVT.outputCommands.extend(RecoMuonRECO.outputCommands)
RecoMuonFEVT.outputCommands.extend(RecoMuonIsolationFEVT.outputCommands)
RecoMuonRECO.outputCommands.extend(RecoMuonIsolationRECO.outputCommands)
RecoMuonAOD.outputCommands.extend(RecoMuonIsolationAOD.outputCommands)


