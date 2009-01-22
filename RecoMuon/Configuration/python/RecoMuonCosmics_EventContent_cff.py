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
    'keep recoMuons_muons_*_*', 
    'keep recoMuons_STAMuons_*_*', 
    'keep recoMuons_TKMuons_*_*', 
    'keep recoMuons_GLBMuons_*_*', 

    # Splitted tracks
    'keep recoTracks_globalCosmicSplittedMuons_*_*', 
    'keep recoTrackExtras_globalCosmicSplittedMuons_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicSplittedMuons_*_*', 
    'keep recoMuons_splittedMuons_*_*', 

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
    'keep recoMuons_STAMuons1Leg_*_*', 
    'keep recoMuons_GLBMuons1Leg_*_*', 

    
    # cosimic reco in barrel only, based on DT no drift algo
    'keep recoTracks_cosmicMuonsNoDriftBarrelOnly_*_*', 
    'keep recoTrackExtras_cosmicMuonsNoDriftBarrelOnly_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuonsNoDriftBarrelOnly_*_*', 
    'keep recoTracks_globalCosmicMuonsNoDriftBarrelOnly_*_*', 
    'keep recoTrackExtras_globalCosmicMuonsNoDriftBarrelOnly_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicMuonsNoDriftBarrelOnly_*_*', 
    'keep recoMuons_muonsNoDriftBarrelOnly_*_*', 
    'keep recoMuons_STAMuonsNoDriftBarrelOnly_*_*', 
    'keep recoMuons_GLBMuonsNoDriftBarrelOnly_*_*',


    # cosimic reco in endcaps only
    'keep recoTracks_cosmicMuonsEndCapsOnly_*_*', 
    'keep recoTrackExtras_cosmicMuonsEndCapsOnly_*_*', 
    'keep TrackingRecHitsOwned_cosmicMuonsEndCapsOnly_*_*', 
    'keep recoTracks_globalCosmicMuonsEndCapsOnly_*_*', 
    'keep recoTrackExtras_globalCosmicMuonsEndCapsOnly_*_*', 
    'keep TrackingRecHitsOwned_globalCosmicMuonsEndCapsOnly_*_*', 
    'keep recoMuons_muonsEndCapsOnly_*_*',
    'keep recoMuons_STAMuonsEndCapsOnly_*_*',
    'keep recoMuons_GLBMuonsEndCapsOnly_*_*',
    
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
                                           'keep *_CosmicMuonSeedNoDriftBarrelOnly_*_*', 
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


