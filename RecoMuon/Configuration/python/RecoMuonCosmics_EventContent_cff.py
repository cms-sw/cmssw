# The following comments couldn't be translated into the new config version:

# Stand Alone's tracks with extra and hits
#inclusive cosmic reco

# cosimic reco in barrel only

# cosimic reco in endcaps only

# cosimic reco "1 Leg type" in barrel only

# cosimic reco in barrel only, based on DT no drift algo

# lhc like reco in barrel only

# lhc like reco in endcap only

# Global's tracks with extra and hits
#inclusive cosmic reco

# cosimic reco in barrel only

# cosimic reco "1 Leg type" in barrel only

# cosimic reco in barrel only, based on DT no drift algo

# Tracker's Tracks without extra and hits

# Muon Id
#inclusive cosmic reco

# cosimic reco in barrel only

# cosimic reco in endcaps only

# cosimic reco "1 Leg type" in barrel only

# cosimic reco in barrel only, based on DT no drift algo

# Seed
#inclusive cosmic reco

# cosimic reco in barrel only

# cosimic reco in endcaps only

# cosimic reco in barrel only, based on DT no drift algo

# lhc like reco in barrel only

# lhc like reco in endcaps only

import FWCore.ParameterSet.Config as cms

#Add Isolation
from RecoMuon.MuonIsolationProducers.muIsolation_EventContent_cff import *
# AOD content
RecoMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_cosmicMuons_*_*', 
        'keep recoTrackExtras_cosmicMuons_*_*', 
        'keep TrackingRecHitsOwned_cosmicMuons_*_*', 
        'keep recoTracks_cosmicMuonsBarrelOnly_*_*', 
        'keep recoTrackExtras_cosmicMuonsBarrelOnly_*_*', 
        'keep TrackingRecHitsOwned_cosmicMuonsBarrelOnly_*_*', 
        'keep recoTracks_cosmicMuonsEndCapsOnly_*_*', 
        'keep recoTrackExtras_cosmicMuonsEndCapsOnly_*_*', 
        'keep TrackingRecHitsOwned_cosmicMuonsEndCapsOnly_*_*', 
        'keep recoTracks_cosmicMuons1LegBarrelOnly_*_*', 
        'keep recoTrackExtras_cosmicMuons1LegBarrelOnly_*_*', 
        'keep TrackingRecHitsOwned_cosmicMuons1LegBarrelOnly_*_*', 
        'keep recoTracks_cosmicMuonsNoDriftBarrelOnly_*_*', 
        'keep recoTrackExtras_cosmicMuonsNoDriftBarrelOnly_*_*', 
        'keep TrackingRecHitsOwned_cosmicMuonsNoDriftBarrelOnly_*_*', 
        'keep recoTracks_lhcStandAloneMuonsBarrelOnly_*_*', 
        'keep recoTrackExtras_lhcStandAloneMuonsBarrelOnly_*_*', 
        'keep TrackingRecHitsOwned_lhcStandAloneMuonsBarrelOnly_*_*', 
        'keep recoTracks_lhcStandAloneMuonsEndCapsOnly_*_*', 
        'keep recoTrackExtras_lhcStandAloneMuonsEndCapsOnly_*_*', 
        'keep TrackingRecHitsOwned_lhcStandAloneMuonsEndCapsOnly_*_*', 
        'keep recoTracks_globalCosmicMuons_*_*', 
        'keep recoTrackExtras_globalCosmicMuons_*_*', 
        'keep TrackingRecHitsOwned_globalCosmicMuons_*_*', 
        'keep recoTracks_globalCosmicMuonsBarrelOnly_*_*', 
        'keep recoTrackExtras_globalCosmicMuonsBarrelOnly_*_*', 
        'keep TrackingRecHitsOwned_globalCosmicMuonsBarrelOnly_*_*', 
        'keep recoTracks_globalCosmicMuons1LegBarrelOnly_*_*', 
        'keep recoTrackExtras_globalCosmicMuons1LegBarrelOnly_*_*', 
        'keep TrackingRecHitsOwned_globalCosmicMuons1LegBarrelOnly_*_*', 
        'keep recoTracks_globalCosmicMuonsNoDriftBarrelOnly_*_*', 
        'keep recoTrackExtras_globalCosmicMuonsNoDriftBarrelOnly_*_*', 
        'keep TrackingRecHitsOwned_globalCosmicMuonsNoDriftBarrelOnly_*_*', 
        'keep recoTracks_ctfWithMaterialTracksP5_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoMuons_STAMuons_*_*', 
        'keep recoMuons_TKMuons_*_*', 
        'keep recoMuons_GLBMuons_*_*', 
        'keep recoMuons_muonsBarrelOnly_*_*', 
        'keep recoMuons_STAMuonsBarrelOnly_*_*', 
        'keep recoMuons_GLBMuonsBarrelOnly_*_*', 
        'keep recoMuons_STAMuonsEndCapsOnly_*_*', 
        'keep recoMuons_muons1LegBarrelOnly_*_*', 
        'keep recoMuons_STAMuons1LegBarrelOnly_*_*', 
        'keep recoMuons_GLBMuons1LegBarrelOnly_*_*', 
        'keep recoMuons_muonsNoDriftBarrelOnly_*_*', 
        'keep recoMuons_STAMuonsNoDriftBarrelOnly_*_*', 
        'keep recoMuons_GLBMuonsNoDriftBarrelOnly_*_*')
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


