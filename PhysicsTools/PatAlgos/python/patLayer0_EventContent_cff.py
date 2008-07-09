# The following comments couldn't be translated into the new config version:

# to allow embedding
#"keep *_iterativeCone5GenJetsPt10_*_*",

# for Layer 1 MET, to be fixed
# for Layer 1 MET, to be fixed
import FWCore.ParameterSet.Config as cms

patLayer0EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmTriggerResults_TriggerResults_*_*', 
        'keep *_genParticles_*_*', 
        'keep recoJetTags_*_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep *_caloTowers_*_*', 
        'keep CaloTowersSorted_towerMaker_*_*', 
        'keep *_layer0MuonIsolations_*_*', 
        'keep *_layer0ElectronIsolation_*_*', 
        'keep *_layer0TauIsolations_*_*', 
        'keep *_layer0PhotonIsolations_*_*', 
        'keep *_layer0JetCorrFactors_*_*', 
        'keep recoGenJetsedmAssociation_*_*_*', 
        'keep *_iterativeCone5GenJets_*_*', 
        'keep *_genMet_*_*', 
        'keep recoMuons_muons__*', 
        'keep recoMuons_paramMuons_ParamGlobalMuons_*', 
        'keep recoGenParticlesedmAssociation_*_*_*', 
        'keep floatedmValueMap_*_*_*', 
        'keep *_electronId_*_*', 
        'keep *_electronIdRobust_*_*', 
        'keep *_jetFlavourAssociation_*_*', 
        'keep *_layer0JetTracksAssociator_*_*', 
        'keep *_layer0JetCharge_*_*', 
        'keep *_layer0BTags_*_*', 
        'keep *_layer0TagInfos_*_*', 
        'keep recoTrackIPTagInfos_*_*_*', 
        'keep recoSoftLeptonTagInfos_*_*_*', 
        'keep recoSecondaryVertexTagInfos_*_*_*', 
        'keep *_allLayer0Photons_*_*', 
        'keep *_allLayer0Electrons_*_*', 
        'keep *_allLayer0Muons_*_*', 
        'keep *_allLayer0Taus_*_*', 
        'keep *_allLayer0Jets_*_*', 
        'keep *_allLayer0PFJets_*_*', 
        'keep *_allLayer0METs_*_*', 
        'drop recoCandidateedmRefToBaseedmValueMap_allLayer0Photons_*_*', 
        'drop recoCandidateedmRefToBaseedmValueMap_allLayer0Electrons_*_*', 
        'drop recoCandidateedmRefToBaseedmValueMap_allLayer0Muons_*_*', 
        'drop recoCandidateedmRefToBaseedmValueMap_allLayer0Taus_*_*', 
        'drop recoCandidateedmRefToBaseedmValueMap_allLayer0Jets_*_*', 
        'drop recoCandidateedmRefToBaseedmValueMap_allLayer0METs_*_*')
)

