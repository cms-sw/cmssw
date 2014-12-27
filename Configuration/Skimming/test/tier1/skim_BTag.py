# Auto generated configuration file
# using:
# Revision: 1.303.2.7
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v
# with command line options: skim -s SKIM:LogError+EXOHSCP --data --conditions FT_R_42_V13A::All --python_filenam skim_BTag.py --magField AutoFromDBCurrent --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('SKIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.Skims_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('skim_RECO.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.303.2.7 $'),
    annotation = cms.untracked.string('skim nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

# Additional output definition
process.SKIMStreamEXOHSCP = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('EXOHSCPPath')
    ),
    outputCommands = cms.untracked.vstring('drop *',
        'keep GenEventInfoProduct_generator_*_*',
        'keep L1GlobalTriggerReadoutRecord_*_*_*',
        'keep recoVertexs_offlinePrimaryVertices_*_*',
        'keep recoMuons_muonsSkim_*_*',
        'keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*',
        'keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*',
        'keep recoTracks_generalTracksSkim_*_*',
        'keep recoTrackExtras_generalTracksSkim_*_*',
        'keep TrackingRecHitsOwned_generalTracksSkim_*_*',
        'keep *_dt1DRecHits_*_*',
        'keep *_dt4DSegments_*_*',
        'keep *_csc2DRecHits_*_*',
        'keep *_cscSegments_*_*',
        'keep *_rpcRecHits_*_*',
        'keep recoTracks_standAloneMuons_*_*',
        'keep recoTrackExtras_standAloneMuons_*_*',
        'keep TrackingRecHitsOwned_standAloneMuons_*_*',
        'keep recoTracks_globalMuons_*_*',
        'keep recoTrackExtras_globalMuons_*_*',
        'keep TrackingRecHitsOwned_globalMuons_*_*',
        'keep EcalRecHitsSorted_reducedHSCPEcalRecHitsEB_*_*',
        'keep EcalRecHitsSorted_reducedHSCPEcalRecHitsEE_*_*',
        'keep HBHERecHitsSorted_reducedHSCPhbhereco__*',
        'keep edmTriggerResults_TriggerResults__*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep *_HSCPIsolation01__*',
        'keep *_HSCPIsolation03__*',
        'keep *_HSCPIsolation05__*',
        'keep recoPFJets_ak4PFJets__*',
        'keep recoPFMETs_pfMet__*',
        'keep recoBeamSpot_offlineBeamSpot__*'),
    fileName = cms.untracked.string('EXOHSCP.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EXOHSCP'),
        dataTier = cms.untracked.string('USER')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880)
)
process.SKIMStreamLogError = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathlogerror')
    ),
    outputCommands = (cms.untracked.vstring('drop *', 'drop *', 'keep  FEDRawDataCollection_rawDataCollector_*_*', 'keep  FEDRawDataCollection_source_*_*', 'keep  FEDRawDataCollection_rawDataCollector_*_*', 'keep  FEDRawDataCollection_source_*_*', 'drop *_hlt*_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep FEDRawDataCollection_rawDataCollector_*_*', 'keep FEDRawDataCollection_source_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep DetIdedmEDCollection_siStripDigis_*_*', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_dt1DCosmicRecHits_*_*', 'keep *_dt4DCosmicSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep HBHERecHitsSorted_hbherecoMB_*_*', 'keep HORecHitsSorted_horecoMB_*_*', 'keep HFRecHitsSorted_hfrecoMB_*_*', 'keep ZDCDataFramesSorted_*Digis_*_*', 'keep ZDCRecHitsSorted_*_*_*', 'keep *_reducedHcalRecHits_*_*', 'keep *_castorreco_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalCompactTrigPrim_*_*', 'keep *_ecalTPSkim_*_*', 'keep *_selectDigi_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsEE_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsEB_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsES_*_*', 'keep *_hybridSuperClusters_*_*', 'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 'keep *_multi5x5BasicClusters_*_*', 'keep recoSuperClusters_multi5x5SuperClusters_*_*', 'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*', 'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*', 'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*', 'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*', 'drop recoClusterShapes_*_*_*', 'drop recoBasicClustersToOnerecoClusterShapesAssociation_*_*_*', 'drop recoBasicClusters_multi5x5BasicClusters_multi5x5BarrelBasicClusters_*', 'drop recoSuperClusters_multi5x5SuperClusters_multi5x5BarrelSuperClusters_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_electronMergedSeeds_*_*', 'keep recoGsfTracks_electronGsfTracks_*_*', 'keep recoGsfTrackExtras_electronGsfTracks_*_*', 'keep recoTrackExtras_electronGsfTracks_*_*', 'keep TrackingRecHitsOwned_electronGsfTracks_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoTrackExtras_generalTracks_*_*', 'keep TrackingRecHitsOwned_generalTracks_*_*', 'keep recoTracks_beamhaloTracks_*_*', 'keep recoTrackExtras_beamhaloTracks_*_*', 'keep TrackingRecHitsOwned_beamhaloTracks_*_*', 'keep recoTracks_regionalCosmicTracks_*_*', 'keep recoTrackExtras_regionalCosmicTracks_*_*', 'keep TrackingRecHitsOwned_regionalCosmicTracks_*_*', 'keep recoTracks_rsWithMaterialTracks_*_*', 'keep recoTrackExtras_rsWithMaterialTracks_*_*', 'keep TrackingRecHitsOwned_rsWithMaterialTracks_*_*', 'keep *_ctfPixelLess_*_*', 'keep *_dedxTruncated40_*_*', 'keep *_dedxDiscrimASmi_*_*', 'keep *_dedxHarmonic2_*_*', 'keep *_trackExtrapolator_*_*', 'keep *_kt4CaloJets_*_*', 'keep *_kt6CaloJets_*_*', 'keep *_ak4CaloJets_*_*', 'keep *_ak7CaloJets_*_*', 'keep *_iterativeCone5CaloJets_*_*', 'keep *_iterativeCone15CaloJets_*_*', 'keep *_kt4PFJets_*_*', 'keep *_kt6PFJets_*_*', 'keep *_ak4PFJets_*_*', 'keep *_ak7PFJets_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_JetPlusTrackZSPCorJetAntiKt5_*_*', 'keep *_ak4TrackJets_*_*', 'keep *_kt4TrackJets_*_*', 'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_CastorTowerReco_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep *_ak4JetTracksAssociatorAtVertex_*_*', 'keep *_ak4JetTracksAssociatorAtCaloFace_*_*', 'keep *_ak5JetExtender_*_*', 'keep *_ak7JetTracksAssociatorAtVertex_*_*', 'keep *_ak7JetTracksAssociatorAtCaloFace_*_*', 'keep *_ak7JetExtender_*_*', 'keep *_ak4JetID_*_*', 'keep *_ak7JetID_*_*', 'keep *_ic5JetID_*_*', 'keep *_kt4JetID_*_*', 'keep *_kt6JetID_*_*', 'keep *_ak7CastorJets_*_*', 'keep *_ak7CastorJetID_*_*', 'keep recoCaloMETs_met_*_*', 'keep recoCaloMETs_metNoHF_*_*', 'keep recoCaloMETs_metHO_*_*', 'keep recoCaloMETs_corMetGlobalMuons_*_*', 'keep recoCaloMETs_metNoHFHO_*_*', 'keep recoCaloMETs_metOptHO_*_*', 'keep recoCaloMETs_metOpt_*_*', 'keep recoCaloMETs_metOptNoHFHO_*_*', 'keep recoCaloMETs_metOptNoHF_*_*', 'keep recoMETs_htMetAK5_*_*', 'keep recoMETs_htMetAK7_*_*', 'keep recoMETs_htMetIC5_*_*', 'keep recoMETs_htMetKT4_*_*', 'keep recoMETs_htMetKT6_*_*', 'keep recoMETs_tcMet_*_*', 'keep recoMETs_tcMetWithPFclusters_*_*', 'keep recoPFMETs_pfMet_*_*', 'keep recoMuonMETCorrectionDataedmValueMap_muonMETValueMapProducer_*_*', 'keep recoMuonMETCorrectionDataedmValueMap_muonTCMETValueMapProducer_*_*', 'keep recoHcalNoiseRBXs_hcalnoise_*_*', 'keep HcalNoiseSummary_hcalnoise_*_*', 'keep *HaloData_*_*_*', 'keep *BeamHaloSummary_BeamHaloSummary_*_*', 'keep *_MuonSeed_*_*', 'keep *_ancientMuonSeed_*_*', 'keep *_mergedStandAloneMuonSeeds_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep TrackingRecHitsOwned_tevMuons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_CosmicMuonSeed_*_*', 'keep recoTrackExtras_cosmicMuons_*_*', 'keep TrackingRecHitsOwned_cosmicMuons_*_*', 'keep recoTrackExtras_globalCosmicMuons_*_*', 'keep TrackingRecHitsOwned_globalCosmicMuons_*_*', 'keep recoTrackExtras_cosmicMuons1Leg_*_*', 'keep TrackingRecHitsOwned_cosmicMuons1Leg_*_*', 'keep recoTrackExtras_globalCosmicMuons1Leg_*_*', 'keep TrackingRecHitsOwned_globalCosmicMuons1Leg_*_*', 'keep recoTracks_cosmicsVetoTracks_*_*', 'keep *_SETMuonSeed_*_*', 'keep recoTracks_standAloneSETMuons_*_*', 'keep recoTrackExtras_standAloneSETMuons_*_*', 'keep TrackingRecHitsOwned_standAloneSETMuons_*_*', 'keep recoTracks_globalSETMuons_*_*', 'keep recoTrackExtras_globalSETMuons_*_*', 'keep TrackingRecHitsOwned_globalSETMuons_*_*', 'keep recoMuons_muonsWithSET_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep recoTracks_tevMuons_*_*', 'keep recoTrackExtras_tevMuons_*_*', 'keep recoTracksToOnerecoTracksAssociation_tevMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoMuons_muons_*_*', 'keep booledmValueMap_muid*_*_*', 'keep recoMuonTimeExtraedmValueMap_muons_*_*', 'keep *_muonShowerInformation_*_*', 'keep recoTracks_cosmicMuons_*_*', 'keep recoTracks_globalCosmicMuons_*_*', 'keep recoMuons_muonsFromCosmics_*_*', 'keep recoTracks_cosmicMuons1Leg_*_*', 'keep recoTracks_globalCosmicMuons1Leg_*_*', 'keep recoMuons_muonsFromCosmics1Leg_*_*', 'keep recoMuonCosmicCompatibilityedmValueMap_cosmicsVeto_*_*', 'keep uintedmValueMap_cosmicsVeto_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_ghostTrackVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_simpleSecondaryVertexHighEffBJetTags_*_*', 'keep *_simpleSecondaryVertexHighPurBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_ghostTrackBJetTags_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronCands_*_*', 'keep *_softPFElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softElectronByIP3dBJetTags_*_*', 'keep *_softElectronByPtBJetTags_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonByIP3dBJetTags_*_*', 'keep *_softMuonByPtBJetTags_*_*', 'keep *_combinedMVABJetTags_*_*', 'keep *_ak4PFJetsRecoTauPiZeros_*_*', 'keep *_hpsPFTauProducer*_*_*', 'keep *_hpsPFTauDiscrimination*_*_*', 'keep *_shrinkingConePFTauProducer*_*_*', 'keep *_shrinkingConePFTauDiscrimination*_*_*', 'keep *_hpsTancTaus_*_*', 'keep *_hpsTancTausDiscrimination*_*_*', 'keep *_TCTauJetPlusTrackZSPCorJetAntiKt5_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep recoCaloTaus_caloRecoTauProducer*_*_*', 'keep *_caloRecoTauDiscrimination*_*_*', 'keep  *_offlinePrimaryVertices_*_*', 'keep  *_offlinePrimaryVerticesWithBS_*_*', 'keep  *_offlinePrimaryVerticesFromCosmicTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep *_generalV0Candidates_*_*', 'keep recoGsfElectronCores_gsfElectronCores_*_*', 'keep recoGsfElectrons_gsfElectrons_*_*', 'keep floatedmValueMap_eidRobustLoose_*_*', 'keep floatedmValueMap_eidRobustTight_*_*', 'keep floatedmValueMap_eidRobustHighEnergy_*_*', 'keep floatedmValueMap_eidLoose_*_*', 'keep floatedmValueMap_eidTight_*_*', 'keep recoPhotons_photons_*_*', 'keep recoPhotonCores_photonCore_*_*', 'keep recoConversions_conversions_*_*', 'drop *_conversions_uncleanedConversions_*', 'keep recoConversions_allConversions_*_*', 'keep recoTracks_ckfOutInTracksFromConversions_*_*', 'keep recoTracks_ckfInOutTracksFromConversions_*_*', 'keep recoTrackExtras_ckfOutInTracksFromConversions_*_*', 'keep recoTrackExtras_ckfInOutTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfOutInTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfInOutTracksFromConversions_*_*', 'keep *_PhotonIDProd_*_*', 'keep *_hfRecoEcalCandidate_*_*', 'keep *_hfEMClusters_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*')+cms.untracked.vstring('keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHFEM_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHFHAD_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitHCAL_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*', 'keep recoPFClusters_particleFlowClusterECAL_*_*', 'keep recoPFClusters_particleFlowClusterHCAL_*_*', 'keep recoPFClusters_particleFlowClusterPS_*_*', 'keep recoPFBlocks_particleFlowBlock_*_*', 'keep recoPFCandidates_particleFlow_*_*', 'keep recoPFDisplacedVertexs_particleFlowDisplacedVertex_*_*', 'keep *_pfElectronTranslator_*_*', 'keep *_pfPhotonTranslator_*_*', 'keep *_trackerDrivenElectronSeeds_preid_*', 'keep *_offlineBeamSpot_*_*', 'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtTriggerMenuLite_*_*', 'keep *_conditionsInEdm_*_*', 'keep *_l1extraParticles_*_*', 'keep L1MuGMTReadoutCollection_gtDigis_*_*', 'keep L1GctEmCand*_gctDigis_*_*', 'keep L1GctJetCand*_gctDigis_*_*', 'keep L1GctEtHad*_gctDigis_*_*', 'keep L1GctEtMiss*_gctDigis_*_*', 'keep L1GctEtTotal*_gctDigis_*_*', 'keep L1GctHtMiss*_gctDigis_*_*', 'keep L1GctJetCounts*_gctDigis_*_*', 'keep L1GctHFRingEtSums*_gctDigis_*_*', 'keep L1GctHFBitCounts*_gctDigis_*_*', 'keep LumiDetails_lumiProducer_*_*', 'keep LumiSummary_lumiProducer_*_*', 'drop *_hlt*_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep L1AcceptBunchCrossings_*_*_*', 'keep L1TriggerScalerss_*_*_*', 'keep Level1TriggerScalerss_*_*_*', 'keep LumiScalerss_*_*_*', 'keep BeamSpotOnlines_*_*_*', 'keep DcsStatuss_*_*_*', 'keep *_logErrorHarvester_*_*', 'drop *_MEtoEDMConverter_*_*', 'drop *_*_*_SKIM')),
    fileName = cms.untracked.string('LogError.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('LogError'),
        dataTier = cms.untracked.string('RAW-RECO')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880)
)

# Other statements
process.GlobalTag.globaltag = 'FT_R_42_V13A::All'

# Path and EndPath definitions
process.SKIMStreamEXOHSCPOutPath = cms.EndPath(process.SKIMStreamEXOHSCP)
process.SKIMStreamLogErrorOutPath = cms.EndPath(process.SKIMStreamLogError)

# Schedule definition
process.schedule = cms.Schedule(process.pathlogerror,process.EXOHSCPPath,process.SKIMStreamEXOHSCPOutPath,process.SKIMStreamLogErrorOutPath)
