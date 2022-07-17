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
    outputCommands = cms.untracked.vstring(#'keep *_muons_*_*',
                                           'keep recoMuons_muons_*_*',
                                           'keep booledmValueMap_muons_*_*',
                                           'keep doubleedmValueMap_muons_muPFMean*_*',
                                           'keep doubleedmValueMap_muons_muPFSum*_*',
                                           'keep *_muons_muonShowerInformation_*',
                                           'keep recoMuonTimeExtraedmValueMap_muons_*_*',
                                           'keep recoMuonCosmicCompatibilityedmValueMap_muons_*_*',
                                           'keep uintedmValueMap_muons_*_*',
                                           'keep *_particleFlow_muons_*',
                                           #'drop *_muons_muons1stStep2muonsMap_*',
                                           #'drop recoIsoDepositedmValueMap_muons_*_*', #not really used
                                           #'drop doubleedmValueMap_muons_muPFIso*_*', #already inside the muon
                                           # displacedMuons collection
                                           'keep recoMuons_displacedMuons_*_*',
                                           'keep booledmValueMap_displacedMuons_*_*',
                                           'keep doubleedmValueMap_displacedMuons_dispMuPFMean*_*',
                                           'keep doubleedmValueMap_displacedMuons_dispMuPFSum*_*',
                                           'keep recoMuonTimeExtraedmValueMap_displacedMuons_*_*',
                                           'keep uintedmValueMap_displacedMuons_*_*',
                                           'keep *_particleFlow_displacedMuons_*',
                                           # Tracks known by the Muon obj
                                           'keep recoTracks_standAloneMuons_*_*',
                                           'keep recoTrackExtras_standAloneMuons_*_*',
                                           'keep TrackingRecHitsOwned_standAloneMuons_*_*',
                                           'keep recoTracks_globalMuons_*_*',
                                           'keep recoTrackExtras_globalMuons_*_*',
                                           'keep recoTracks_tevMuons_*_*',
                                           'keep recoTrackExtras_tevMuons_*_*',
                                           'keep recoTracks_generalTracks_*_*',
                                           'keep recoTracks_displacedTracks_*_*',
                                           'keep recoTracksToOnerecoTracksAssociation_tevMuons_*_*',
                                           # Displaced Global Muons
                                           'keep recoTracks_displacedGlobalMuons_*_*',
                                           'keep recoTrackExtras_displacedGlobalMuons_*_*',
                                           'keep TrackingRecHitsOwned_displacedGlobalMuons_*_*',
                                           # Cosmics
                                           'keep recoTracks_cosmicMuons_*_*',
                                           'keep recoMuons_muonsFromCosmics_*_*',
                                           # Cosmics 1 leg
                                           'keep recoTracks_cosmicMuons1Leg_*_*',
                                           'keep recoMuons_muonsFromCosmics1Leg_*_*',
                                           # Additional tracks
                                           'keep recoTracks_refittedStandAloneMuons_*_*',
                                           'keep recoTrackExtras_refittedStandAloneMuons_*_*',
                                           'keep TrackingRecHitsOwned_refittedStandAloneMuons_*_*',
                                           'keep recoTracks_displacedStandAloneMuons__*',
                                           'keep recoTrackExtras_displacedStandAloneMuons_*_*',
                                           'keep TrackingRecHitsOwned_displacedStandAloneMuons_*_*',
                                           'keep *_muonReducedTrackExtras_*_*',
                                           'keep *_displacedMuonReducedTrackExtras_*_*')
)
RecoMuonAOD.outputCommands.extend(RecoMuonIsolationAOD.outputCommands)

# RECO content
RecoMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_MuonSeed_*_*',
                                           'keep *_ancientMuonSeed_*_*',
                                           'keep *_displacedMuonSeeds_*_*',
                                           'keep TrackingRecHitsOwned_globalMuons_*_*',
                                           'keep TrackingRecHitsOwned_tevMuons_*_*',
                                           # Cosmics
                                           'keep *_CosmicMuonSeed_*_*',
                                           'keep recoTrackExtras_cosmicMuons_*_*',
                                           'keep TrackingRecHitsOwned_cosmicMuons_*_*',
                                           'keep recoTrackExtras_cosmicMuons1Leg_*_*',
                                           'keep TrackingRecHitsOwned_cosmicMuons1Leg_*_*',
                                           'keep recoTracks_cosmicsVetoTracks_*_*')
)
RecoMuonRECO.outputCommands.extend(RecoMuonAOD.outputCommands)
RecoMuonRECO.outputCommands.extend(RecoMuonIsolationRECO.outputCommands)

# Full Event content
RecoMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoMuonFEVT.outputCommands.extend(RecoMuonRECO.outputCommands)
RecoMuonFEVT.outputCommands.extend(RecoMuonIsolationFEVT.outputCommands)
