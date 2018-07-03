import FWCore.ParameterSet.Config as cms

patEventContentNoCleaning = [
    'keep *_selectedPatPhotons*_*_*',
    'keep *_selectedPatOOTPhotons*_*_*',
    'keep *_selectedPatElectrons*_*_*',
    'keep *_selectedPatMuons*_*_*',
    'keep *_selectedPatTaus*_*_*',
    'keep *_selectedPatJets*_*_*',    
    'drop *_*PF_caloTowers_*',
    'drop *_*JPT_pfCandidates_*',
    'drop *_*Calo_pfCandidates_*',
    'keep *_patMETs*_*_*',
    'keep *_selectedPatPFParticles*_*_*',
    'keep *_selectedPatTrackCands*_*_*'
]

patEventContent = [
    'keep *_selectedPatJets*_*_*',           ## keep refactorized pat jet elements
    'drop patJets_selectedPatJets*_*_*',     ## drop the actual selected pat jets, they're redundant
    'drop *_selectedPatJets_pfCandidates_*', ## drop for default patJets which are CaloJets
    'drop *_*PF_caloTowers_*',               ## drop collections not needed for the corresponding jet types
    'drop *_*JPT_pfCandidates_*',            ## drop collections not needed for the corresponding jet types
    'drop *_*Calo_pfCandidates_*',           ## drop collections not needed for the corresponding jet types
    'keep *_cleanPatPhotons*_*_*',
    'keep *_cleanPatElectrons*_*_*',
    'keep *_cleanPatMuons*_*_*',
    'keep *_cleanPatTaus*_*_*',
    'keep *_cleanPatJets*_*_*',
    'keep *_patMETs*_*_*',
    'keep *_cleanPatHemispheres*_*_*',
    'keep *_cleanPatPFParticles*_*_*',
    'keep *_cleanPatTrackCands*_*_*'
]

patExtraAodEventContent = [
    # GEN
    'keep recoGenParticles_genParticles*_*_*',
    'keep GenEventInfoProduct_*_*_*',
    'keep GenRunInfoProduct_*_*_*',
    # RECO
    'keep recoTracks_generalTracks*_*_*',
    'keep *_towerMaker_*_*',
    'keep *_offlineBeamSpot_*_*',
    'keep *_offlinePrimaryVertices*_*_*',
    # TRIGGER
    'keep edmTriggerResults_TriggerResults*_*_*',
    'keep *_hltTriggerSummaryAOD_*_*',
    'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
    # COND
    'keep edmConditionsIn*Block_conditionsInEdm_*_*'
]

patTriggerEventContent = [
    'keep patTriggerAlgorithms_patTrigger_*_*',
    'keep patTriggerConditions_patTrigger_*_*',
    'keep patTriggerObjects_patTrigger_*_*',
    'keep patTriggerFilters_patTrigger_*_*',
    'keep patTriggerPaths_patTrigger_*_*',
    'keep *_patTriggerEvent_*_*'
]
patTriggerStandAloneEventContent = [
    'keep patTriggerObjectStandAlones_patTrigger_*_*',
    'keep patTriggerObjectStandAlonesedmAssociation_*_*_*'
]
patTriggerL1RefsEventContent = [
    'keep *_l1extraParticles_*_*',
    'keep *_gctDigis_*_*'
]

patEventContentTriggerMatch = [
    'keep *_*PatPhotons*TriggerMatch_*_*',
    'keep *_*PatElectrons*TriggerMatch_*_*',
    'keep *_*PatMuons*TriggerMatch_*_*',
    'keep *_*PatTaus*TriggerMatch_*_*',
    'keep *_*PatJets*TriggerMatch_*_*',
    'keep *_patMETs*TriggerMatch_*_*'
]

patHiEventContent = [
    'keep patPhotons_selected*_*_*',
    'keep patMuons_selected*_*_*',
    'keep patJets_selected*_*_*',
    'keep patHeavyIon_heavyIon_*_*'
]

patHiExtraAodEventContent = [
    'keep recoGenParticles_hiGenParticles*_*_*',
    'keep recoGenJets_iterativeCone5HiGenJets*_*_*', # until a better solution
    'keep recoTracks_hiSelectedTracks*_*_*'
]
