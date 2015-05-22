
import FWCore.ParameterSet.Config as cms

import copy 

bare = [
    'drop recoCaloTau*_*_*_*',
    'drop recoPFTau*_*_*_*',
    'drop recoCaloJet*_*_*_*',
    'drop recoPFJet*_*_*_*',
    'drop recoJPTJets_*_*_*',
    'drop recoTrackJets_*_*_*',
    'drop recoJetIDedmValueMap_*_*_*',
    'drop recoConversions_*_*_*', 
    'drop recoJetedmRefToBaseProdTofloatsAssociationVector_*_*_*',
    'drop recoPreshowerClusters_*_*_*',
    'drop recoMETs_*_*_*',
    'drop recoPFMETs_*_*_*',
    'drop recoCaloMETs_*_*_*',
    # caloMET can always be useful for understanding fake MET 
    'keep recoCaloMETs_corMetGlobalMuons_*_*',
    'drop *_genMetCalo_*_*',
    'drop *_genMetCaloAndNonPrompt_*_*',
    'drop *_tevMuons_*_*',
    'drop *_generalV0Candidates_*_*',
    'drop *_*TracksFromConversions_*_*',
    'drop recoPhoton*_*_*_*',
    'drop *_muIsoDeposit*_*_*',
    'drop recoMuonMETCorrectionDataedmValueMap_*_*_*',
    'drop *_*JetTracksAssociator*_*_*',
    'drop *_*JetExtender_*_*',
    'drop recoSoftLeptonTagInfos_*_*_*',
    'drop *_impactParameterTagInfos_*_*',
    'drop *_towerMaker_*_*',
    'drop *_sisCone*_*_*',
    'drop *_PhotonIDProd_*_*',
    'drop recoHFEMClusterShapes_*_*_*', 
    'drop recoCaloClustersToOnereco*_*_*_*',
    'drop EcalRecHitsSorted_*_*_*',
    # the next 2 are needed for fake MET event cleaning (RA2 filters)
    'keep EcalRecHitsSorted_reducedEcalRecHitsEB_*_*',
    'keep EcalRecHitsSorted_reducedEcalRecHitsEE_*_*',
    # 'keep EcalTriggerPrimitiveDigisSorted_ecalTPSkim_*_*',
    'drop recoCaloClusters_*_*_*',
    # needed somewhere in PAT. and could be useful in the future. 
    #        'drop *_softPFElectrons_*_*',
    'drop *_particleFlow_electrons_*',
    'drop recoPreshowerClusterShapes_*_*_*',
    # needed in PAT by allLayer1Electrons - dunno why:
    #        'drop *_gsfElectronCores_*_*',
    'drop *_hfRecoEcalCandidate_*_*',
    'drop recoSuperClusters_*_*_*',
    'keep *_pfElectronTranslator_*_*',
    'keep recoSuperClusters_corrected*_*_*',
    'keep *_TriggerResults_*_*',
    'keep *_hltTriggerSummaryAOD_*_*',
    'keep *_lumiProducer_*_*'
    ]

V2 = copy.copy( bare )
V2.extend(
    [
    'keep GenEventInfoProduct_*_*_*',
    'keep *_ak5GenJets_*_*',
    'keep *_ak5CaloJets_*_*',
    'keep *_ak5JetID_*_*',
    'keep *_ak5JetExtender_*_*',
    'keep *_ak7GenJets_*_*',
    'keep *_ak7CaloJets_*_*',
    'keep *_ak7JetID_*_*',
    'keep *_ak7JetExtender_*_*',
    #------- PFJet collections --------
    'keep *_kt6PFJets_rho_*',
    'keep *_kt6PFJets_sigma_*',
    'keep *_ak5PFJets_*_*',        'keep *_ak7PFJets_*_*',
    #------- Trigger collections ------
    'keep edmTriggerResults_TriggerResults_*_*',
    'keep *_hltTriggerSummaryAOD_*_*',
    'keep L1GlobalTriggerObjectMapRecord_*_*_*',
    'keep L1GlobalTriggerReadoutRecord_*_*_*',
    #------- Various collections ------
    'keep *_EventAuxilary_*_*',
    'keep *_offlinePrimaryVertices_*_*',
    'keep *_offlinePrimaryVerticesWithBS_*_*',
    #------- MET collections ----------
    'keep *_met_*_*',
    'keep *_pfMet_*_*'
    ]
    )

V3 = copy.copy( bare )
V3.extend(
    [
    'drop *Castor*_*_*_*',
    'keep recoCaloClusters_hybridSuperClusters_hybridBarrelBasicClusters_*',
    'keep recoCaloClusters_multi5x5BasicClusters_multi5x5EndcapBasicClusters_*',
    'keep recoCaloClusters_hybridSuperClusters_uncleanOnlyHybridBarrelBasicClusters_*',
    'keep recoSuperClusters_hybridSuperClusters_uncleanOnlyHybridSuperClusters_*',
    'keep recoCaloClusters_pfPhotonTranslator_pfphot_*',
    'keep recoTracks_tevMuons_default_*',
    'keep recoTracks_tevMuons_dyt_*',
    'keep recoTracks_tevMuons_firstHit_*',
    'keep recoTracks_tevMuons_picky_*',
    'keep recoTrackExtras_tevMuons_default_*',
    'keep recoTrackExtras_tevMuons_dyt_*',
    'keep recoTrackExtras_tevMuons_firstHit_*',
    'keep recoTrackExtras_tevMuons_picky_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_default_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_dyt_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_firstHit_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_picky_*',
    'keep *_ak7CaloJets_*_*',
    'keep recoPhotonCores_photonCore__*',
    'keep recoPhotons_pfPhotonTranslator_pfphot_*',
    'keep recoPhotons_photons__*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDLoose_*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDLooseEM_*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDTight_*',
    'keep recoPreshowerClusters_pfPhotonTranslator_pfphot_*',
    'keep recoSuperClusters_pfPhotonTranslator_pfphot_*',
    ]
    )


V4 = copy.copy( bare )
V4.extend(
    [
    'drop *Castor*_*_*_*',
    'keep recoCaloClusters_hybridSuperClusters_hybridBarrelBasicClusters_*',
    'keep recoCaloClusters_multi5x5BasicClusters_multi5x5EndcapBasicClusters_*',
    'keep recoCaloClusters_hybridSuperClusters_uncleanOnlyHybridBarrelBasicClusters_*',
    'keep recoSuperClusters_hybridSuperClusters_uncleanOnlyHybridSuperClusters_*',
    'keep recoCaloClusters_pfPhotonTranslator_pfphot_*',
    'keep recoTracks_tevMuons_default_*',
    'keep recoTracks_tevMuons_dyt_*',
    'keep recoTracks_tevMuons_firstHit_*',
    'keep recoTracks_tevMuons_picky_*',
    'keep recoTrackExtras_tevMuons_default_*',
    'keep recoTrackExtras_tevMuons_dyt_*',
    'keep recoTrackExtras_tevMuons_firstHit_*',
    'keep recoTrackExtras_tevMuons_picky_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_default_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_dyt_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_firstHit_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_picky_*',
    'keep *_ak7CaloJets_*_*',
    'keep recoPhotonCores_photonCore__*',
    'keep recoPhotons_pfPhotonTranslator_pfphot_*',
    'keep recoPhotons_photons__*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDLoose_*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDLooseEM_*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDTight_*',
    'keep recoPreshowerClusters_pfPhotonTranslator_pfphot_*',
    'keep recoSuperClusters_pfPhotonTranslator_pfphot_*',    
    'keep recoIsoDepositedmValueMap_muIsoDepositCalByAssociatorTowers_ecal_*',
    'keep recoIsoDepositedmValueMap_muIsoDepositCalByAssociatorTowers_hcal_*',
    'keep recoIsoDepositedmValueMap_muIsoDepositCalByAssociatorTowers_ho_*',
    'keep recoIsoDepositedmValueMap_muIsoDepositJets__*',
    'keep recoIsoDepositedmValueMap_muIsoDepositTk__*',
    'keep EcalRecHitsSorted_reducedEcalRecHitsES__*',
    #just for embedded samples
    'keep *_tmfTracks_*_*'
    ]
    )


V5 = copy.copy( bare )
V5.extend(
    [
    'drop *Castor*_*_*_*',
    'keep recoCaloClusters_hybridSuperClusters_hybridBarrelBasicClusters_*',
    'keep recoCaloClusters_multi5x5BasicClusters_multi5x5EndcapBasicClusters_*',
    'keep recoCaloClusters_multi5x5SuperClusters_multi5x5EndcapBasicClusters_*',
    'keep recoConversions_allConversions__*',
    'keep recoCaloClusters_hybridSuperClusters_uncleanOnlyHybridBarrelBasicClusters_*',
    'keep recoSuperClusters_hybridSuperClusters_uncleanOnlyHybridSuperClusters_*',
    'keep recoCaloClusters_pfPhotonTranslator_pfphot_*',
    'keep recoTracks_tevMuons_default_*',
    'keep recoTracks_tevMuons_dyt_*',
    'keep recoTracks_tevMuons_firstHit_*',
    'keep recoTracks_tevMuons_picky_*',
    'keep recoTrackExtras_tevMuons_default_*',
    'keep recoTrackExtras_tevMuons_dyt_*',
    'keep recoTrackExtras_tevMuons_firstHit_*',
    'keep recoTrackExtras_tevMuons_picky_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_default_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_dyt_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_firstHit_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_picky_*',
    'keep *_ak7CaloJets_*_*',
    'keep double_kt6PFJets_rho_*',
    'keep recoPFJets_ak5PFJets_*_*',
    'keep recoPFMETs_pfMet_*_*',
    'keep recoPhotonCores_photonCore__*',
    'keep recoPhotons_pfPhotonTranslator_pfphot_*',
    'keep recoPhotons_photons__*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDLoose_*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDLooseEM_*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDTight_*',
    'keep recoPreshowerClusters_pfPhotonTranslator_pfphot_*',
    'keep recoSuperClusters_pfPhotonTranslator_pfphot_*',    
    'keep recoIsoDepositedmValueMap_muIsoDepositCalByAssociatorTowers_ecal_*',
    'keep recoIsoDepositedmValueMap_muIsoDepositCalByAssociatorTowers_hcal_*',
    'keep recoIsoDepositedmValueMap_muIsoDepositCalByAssociatorTowers_ho_*',
    'keep recoIsoDepositedmValueMap_muIsoDepositJets__*',
    'keep recoIsoDepositedmValueMap_muIsoDepositTk__*',
    'keep EcalRecHitsSorted_reducedEcalRecHitsES__*',
    #just for old pf embedded samples
    'keep *_tmfTracks_*_*',
    #generalTracksORG needed for new rec-hit embedded samples
    'keep recoTracks_generalTracksORG__EmbeddedRECO',
    #weights needed in rec-hit embedded samples
    'keep double_*_*_Embedded*',
    'keep recoCaloClusters_*_*_*',
    'keep recoSuperClusters_*_*_*',
    'drop recoGenJets_*_*_*'
    ]
    )
