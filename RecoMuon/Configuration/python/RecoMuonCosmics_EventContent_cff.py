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
    'keep recoCaloMuons_calomuons_*_*',

    # Split tracks
    'keep recoTracks_globalCosmicSplitMuons_*_*', 
    'keep recoTrackExtras_globalCosmicSplitMuons_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicSplitMuons_*_*', 
    'keep recoMuons_splitMuons_*_*', 

    # cosmic reco without RPC
    'keep recoTracks_cosmicMuonsNoRPC_*_*', 
    'keep recoTrackExtras_cosmicMuonsNoRPC_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuonsNoRPC_*_*', 
    'keep recoTracks_globalCosmicMuonsNoRPC_*_*', 
    'keep recoTrackExtras_globalCosmicMuonsNoRPC_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicMuonsNoRPC_*_*', 
    'keep recoMuons_muonsNoRPC_*_*', 


    # cosimic reco in barrel only
    'keep recoTracks_cosmicMuonsBarrelOnly_*_*', 
    'keep recoTrackExtras_cosmicMuonsBarrelOnly_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuonsBarrelOnly_*_*', 
    'keep recoTracks_globalCosmicMuonsBarrelOnly_*_*', 
    'keep recoTrackExtras_globalCosmicMuonsBarrelOnly_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicMuonsBarrelOnly_*_*', 
    'keep recoMuons_muonsBarrelOnly_*_*', 
    'keep recoMuons_STAMuonsBarrelOnly_*_*', 
    'keep recoMuons_GLBMuonsBarrelOnly_*_*',
    

    # cosimic reco "1 Leg type" in barrel only
    'keep recoTracks_cosmicMuons1Leg_*_*', 
    'keep recoTrackExtras_cosmicMuons1Leg_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuons1Leg_*_*', 
    'keep recoTracks_globalCosmicMuons1Leg_*_*', 
    'keep recoTrackExtras_globalCosmicMuons1Leg_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicMuons1Leg_*_*', 
    'keep recoMuons_muons1Leg_*_*', 
    
    # cosimic reco with t0 correction in DTs
    'keep recoTracks_cosmicMuonsWitht0Correction_*_*', 
    'keep recoTrackExtras_cosmicMuonsWitht0Correction_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuonsWitht0Correction_*_*', 
    'keep recoTracks_globalCosmicMuonsWitht0Correction_*_*', 
    'keep recoTrackExtras_globalCosmicMuonsWitht0Correction_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicMuonsWitht0Correction_*_*', 
    'keep recoMuons_muonsWitht0Correction_*_*', 

    # cosimic reco in endcaps only
    'keep recoTracks_cosmicMuonsEndCapsOnly_*_*', 
    'keep recoTrackExtras_cosmicMuonsEndCapsOnly_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuonsEndCapsOnly_*_*', 
    'keep recoTracks_globalCosmicMuonsEndCapsOnly_*_*', 
    'keep recoTrackExtras_globalCosmicMuonsEndCapsOnly_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicMuonsEndCapsOnly_*_*', 
    'keep recoMuons_muonsEndCapsOnly_*_*',
    
    # Beam halo in Encaps only
    'keep recoTracks_globalBeamHaloMuonEndCapslOnly_*_*', 
    'keep recoTrackExtras_globalBeamHaloMuonEndCapslOnly_*_*', 
    'keep TrackingRecHitsOwned_globalBeamHaloMuonEndCapslOnly_*_*',
    'keep recoMuons_muonsBeamHaloEndCapsOnly_*_*',
    
    # lhc like reco
    'keep recoTracks_lhcStandAloneMuonsBarrelOnly_*_*', 
    'keep recoTrackExtras_lhcStandAloneMuonsBarrelOnly_*_*',
    'keep TrackingRecHitsOwned_lhcStandAloneMuonsBarrelOnly_*_*', 
    'keep recoMuons_lhcSTAMuonsBarrelOnly_*_*',
    'keep recoTracks_lhcStandAloneMuonsEndCapsOnly_*_*', 
    'keep recoTrackExtras_lhcStandAloneMuonsEndCapsOnly_*_*', 
    'keep TrackingRecHitsOwned_lhcStandAloneMuonsEndCapsOnly_*_*',
    'keep recoMuons_lhcSTAMuonsEndCapsOnly_*_*',

    # Tracker Collections
    'keep recoTracks_ctfWithMaterialTracksP5_*_*', 
    'keep recoTracks_ctfWithMaterialTracksBeamHaloMuon_*_*',
    'keep recoTracks_ctfWithMaterialTracksP5LHCNavigation_*_*')
    )
# RECO content
RecoMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_CosmicMuonSeed_*_*', 
                                           'keep *_CosmicMuonSeedBarrelOnly_*_*', 
                                           'keep *_CosmicMuonSeedEndCapsOnly_*_*', 
                                           'keep *_CosmicMuonSeedWitht0Correction_*_*', 
                                           'keep *_lhcMuonSeedBarrelOnly_*_*', 
                                           'keep *_lhcMuonSeedEndCapsOnly_*_*')
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


