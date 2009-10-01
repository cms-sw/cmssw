# The following comments couldn't be translated into the new config version:

import FWCore.ParameterSet.Config as cms

patEventContentNoLayer1Cleaning = [
    'keep *_selectedLayer1Photons_*_*', 
    'keep *_selectedLayer1Electrons_*_*', 
    'keep *_selectedLayer1Muons_*_*', 
    'keep *_selectedLayer1Taus_*_*', 
    'keep *_selectedLayer1Jets_*_*', 
    'keep *_layer1METs_*_*',
    'keep *_selectedLayer1PFParticles_*_*'
]

patEventContent = [
    'keep *_cleanLayer1Photons_*_*', 
    'keep *_cleanLayer1Electrons_*_*', 
    'keep *_cleanLayer1Muons_*_*', 
    'keep *_cleanLayer1Taus_*_*', 
    'keep *_cleanLayer1Jets_*_*', 
    'keep *_layer1METs_*_*',
    'keep *_cleanLayer1Hemispheres_*_*',
    'keep *_cleanLayer1PFParticles_*_*'
]

patExtraAodEventContent = [
    # GEN
    'keep recoGenParticles_genParticles_*_*',
    'keep GenEventInfoProduct_*_*_*',
    'keep GenRunInfoProduct_*_*_*', 
    # RECO
    'keep recoTracks_generalTracks_*_*', 
    'keep *_towerMaker_*_*',
    'keep *_offlineBeamSpot_*_*',
    'keep *_offlinePrimaryVertices_*_*',
    # TRIGGER
    'keep edmTriggerResults_TriggerResults_*_HLT', 
    'keep *_hltTriggerSummaryAOD_*_*'
]

patTriggerEventContent = [
    'keep patTriggerObjects_patTrigger_*_*',
    'keep patTriggerFilters_patTrigger_*_*',
    'keep patTriggerPaths_patTrigger_*_*',
    'keep patTriggerEvent_patTriggerEvent_*_*'
]
patTriggerStandAloneEventContent = [
    'keep patTriggerObjectStandAlones_patTrigger_*_*',
    'keep patTriggerObjectStandAlonesedmAssociation_*_*_*'
]

patEventContentTriggerMatch = [
    'keep *_cleanLayer1PhotonsTriggerMatch_*_*', 
    'keep *_cleanLayer1ElectronsTriggerMatch_*_*', 
    'keep *_cleanLayer1MuonsTriggerMatch_*_*', 
    'keep *_cleanLayer1TausTriggerMatch_*_*', 
    'keep *_cleanLayer1JetsTriggerMatch_*_*', 
    'keep *_layer1METsTriggerMatch_*_*'
]
