import FWCore.ParameterSet.Config as cms

patLayer0EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmTriggerResults_TriggerResults_*_*', 
        'keep *_genParticles_*_*', 
        'keep recoJetTags_*_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_gsWithMaterialTracks_*_*', 
        'keep *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep *_layer0MuonIsolations_*_*', 
        'keep *_layer0PhotonIsolations_*_*', 
        'keep floatedmValueMap_*_*_*', 
        'keep patJetCorrFactorsedmValueMap_*_*_*', 
        'keep recoGenJetsedmAssociation_*_*_*', 
        'keep recoGenParticlesedmAssociation_*_*_*', 
        'keep recoJetTagsrecoJetTagrecoJetTagsrecoJetTagedmrefhelperFindUsingAdvanceedmRefedmValueMap_*_*_*', 
        'keep recoGsfElectronsrecoElectronIDsuintedmOneToOneedmAssociationMap_electronId_*_*', 
        'keep recoGsfElectronsrecoElectronIDsuintedmOneToOneedmAssociationMap_electronIdRobust_*_*', 
        'keep *_allLayer0Photons_*_*', 
        'keep *_allLayer0Electrons_*_*', 
        'keep *_allLayer0Muons_*_*', 
        'keep *_allLayer0Taus_*_*', 
        'keep *_allLayer0Jets_*_*', 
        'keep *_allLayer0METs_*_*')
)

