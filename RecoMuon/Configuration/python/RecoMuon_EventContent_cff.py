# The following comments couldn't be translated into the new config version:

# Stand Alone's tracks with extra and hits

# Global's tracks with extra and hits

# TeV muons products

# Tracker's Tracks without extra and hits

# Muon Id

# Seed

# Global's tracks with extra and hits

# TeV muons products

import FWCore.ParameterSet.Config as cms

#Add Isolation
from RecoMuon.MuonIsolationProducers.muIsolation_EventContent_cff import *
# AOD content
RecoMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_muons_*_*',
                                           'keep *_*_muons_*',
                                           'drop *_muons_muons1stStep2muonsMap_*',
                                           'drop recoIsoDepositedmValueMap_muons_*_*', #not really used
                                           'drop doubleedmValueMap_muons_muPFIso*_*', #already inside the muon
                                           # Tracks known by the Muon obj
                                           'keep recoTracks_standAloneMuons_*_*', 
                                           'keep recoTrackExtras_standAloneMuons_*_*', 
                                           'keep TrackingRecHitsOwned_standAloneMuons_*_*', 
                                           'keep recoTracks_globalMuons_*_*', 
                                           'keep recoTrackExtras_globalMuons_*_*', 
                                           'keep recoTracks_tevMuons_*_*', 
                                           'keep recoTrackExtras_tevMuons_*_*', 
                                           'keep recoTracks_generalTracks_*_*',
                                           'keep recoTracksToOnerecoTracksAssociation_tevMuons_*_*', 
                                           # Cosmics
                                           'keep recoTracks_cosmicMuons_*_*',
                                           'keep recoTracks_globalCosmicMuons_*_*', 
                                           'keep recoMuons_muonsFromCosmics_*_*',
                                           # Cosmics 1 leg
                                           'keep recoTracks_cosmicMuons1Leg_*_*',
                                           'keep recoTracks_globalCosmicMuons1Leg_*_*', 
                                           'keep recoMuons_muonsFromCosmics1Leg_*_*',
                                           # Additional tracks
                                           'keep recoTracks_refittedStandAloneMuons_*_*', 
                                           'keep recoTrackExtras_refittedStandAloneMuons_*_*', 
                                           'keep TrackingRecHitsOwned_refittedStandAloneMuons_*_*' 
                                           )
)
# RECO content
RecoMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_MuonSeed_*_*',
                                           'keep *_ancientMuonSeed_*_*',
                                           'keep *_mergedStandAloneMuonSeeds_*_*',
                                           'keep TrackingRecHitsOwned_globalMuons_*_*', 
                                           'keep TrackingRecHitsOwned_tevMuons_*_*',
                                           'keep recoCaloMuons_calomuons_*_*',
                                           # Cosmics
                                           'keep *_CosmicMuonSeed_*_*',
                                           'keep recoTrackExtras_cosmicMuons_*_*', 
                                           'keep TrackingRecHitsOwned_cosmicMuons_*_*',
                                           'keep recoTrackExtras_globalCosmicMuons_*_*', 
                                           'keep TrackingRecHitsOwned_globalCosmicMuons_*_*',
                                           'keep recoTrackExtras_cosmicMuons1Leg_*_*', 
                                           'keep TrackingRecHitsOwned_cosmicMuons1Leg_*_*',
                                           'keep recoTrackExtras_globalCosmicMuons1Leg_*_*', 
                                           'keep TrackingRecHitsOwned_globalCosmicMuons1Leg_*_*',
                                           'keep recoTracks_cosmicsVetoTracks_*_*',
                                           # SET muons
                                           'keep *_SETMuonSeed_*_*',
                                           'keep recoTracks_standAloneSETMuons_*_*',
                                           'keep recoTrackExtras_standAloneSETMuons_*_*',
                                           'keep TrackingRecHitsOwned_standAloneSETMuons_*_*',
                                           'keep recoTracks_globalSETMuons_*_*',
                                           'keep recoTrackExtras_globalSETMuons_*_*',
                                           'keep TrackingRecHitsOwned_globalSETMuons_*_*',
                                           'keep recoMuons_muonsWithSET_*_*')
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


