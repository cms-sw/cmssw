import FWCore.ParameterSet.Config as cms

patEventContentNoCleaning = [
    'keep *_selectedPatPhotons_*_*', 
    'keep *_selectedPatElectrons_*_*', 
    'keep *_selectedPatMuons_*_*', 
    'keep *_selectedPatTaus_*_*', 
    'keep *_selectedPatJets*_*_*', 
    'keep *_patMETs*_*_*',
    'keep *_selectedPatPFParticles_*_*'
]

patEventContent = [
    'keep *_cleanPatPhotons_*_*', 
    'keep *_cleanPatElectrons_*_*', 
    'keep *_cleanPatMuons_*_*', 
    'keep *_cleanPatTaus_*_*', 
    'keep *_cleanPat*Jets_*_*', 
    'keep *_patMETs*_*_*',
    'keep *_cleanPatHemispheres_*_*',
    'keep *_cleanPatPFParticles_*_*'
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
    'keep edmTriggerResults_TriggerResults_*_*', 
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
    'keep *_cleanPatPhotonsTriggerMatch_*_*', 
    'keep *_cleanPatElectronsTriggerMatch_*_*', 
    'keep *_cleanPatMuonsTriggerMatch_*_*', 
    'keep *_cleanPatTausTriggerMatch_*_*', 
    'keep *_cleanPatJetsTriggerMatch_*_*', 
    'keep *_layer1METsTriggerMatch_*_*'
]

patHiEventContent = [
    'keep patPhotons_selected*_*_*',
    'keep patMuons_selected*_*_*',
    'keep patJets_selected*_*_*',
    'keep patHeavyIon_heavyIon_*_*'
]

patHiExtraAodEventContent = [
    'keep recoGenParticles_hiGenParticles_*_*',
    'keep recoGenJets_iterativeCone5HiGenJets_*_*', # until a better solution
    'keep recoTracks_hiSelectedTracks_*_*'
]
