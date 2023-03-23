# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: skims -s SKIM:EXOLLPJetHCAL --dasquery=file=/store/relval/CMSSW_12_4_11/DisplacedJet/RECO/124X_dataRun3_Prompt_v4_gtval_RelVal_2022D-v1/2580000/1e9b47ed-c192-4f49-9e3d-6fac61586946.root -n 100 --conditions 120X_mcRun3_2021_realistic_v6 --python_filename=test_EXOLLPJetHCAL_SKIM.py --processName=SKIMEXOLLPJetHCAL --no_exec --data
import FWCore.ParameterSet.Config as cms



process = cms.Process('SKIMEXOLLPJetHCAL')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Skims_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_12_4_11/DisplacedJet/RECO/124X_dataRun3_Prompt_v4_gtval_RelVal_2022D-v1/2580000/1e9b47ed-c192-4f49-9e3d-6fac61586946.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    accelerators = cms.untracked.vstring('*'),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('skims nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string(''),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('skims_SKIM.root'),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
process.SKIMStreamEXOLLPJetHCAL = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('EXOLLPJetHCALPath')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('AOD'),
        filterName = cms.untracked.string('EXOLLPJetHCAL')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('EXOLLPJetHCAL.root'),
    outputCommands = cms.untracked.vstring( (
        'drop *',
        'drop *',
        'keep ClusterSummary_clusterSummaryProducer_*_*',
        'keep *_dt4DSegments_*_*',
        'keep *_dt4DCosmicSegments_*_*',
        'keep *_cscSegments_*_*',
        'keep *_rpcRecHits_*_*',
        'keep *_castorreco_*_*',
        'keep *_reducedHcalRecHits_*_*',
        'keep HcalUnpackerReport_castorDigis_*_*',
        'keep HcalUnpackerReport_hcalDigiAlCaMB_*_*',
        'keep HcalUnpackerReport_hcalDigis_*_*',
        'keep *_selectDigi_*_*',
        'keep EcalRecHitsSorted_reducedEcalRecHitsEB_*_*',
        'keep EcalRecHitsSorted_reducedEcalRecHitsEE_*_*',
        'keep EcalRecHitsSorted_reducedEcalRecHitsES_*_*',
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*',
        'keep recoCaloClusters_hybridSuperClusters_*_*',
        'keep recoSuperClusters_hybridSuperClusters_uncleanOnlyHybridSuperClusters_*',
        'keep recoCaloClusters_multi5x5SuperClusters_multi5x5EndcapBasicClusters_*',
        'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*',
        'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*',
        'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*',
        'keep recoSuperClusters_particleFlowSuperClusterECAL_*_*',
        'keep recoCaloClusters_particleFlowSuperClusterECAL_*_*',
        'keep recoSuperClusters_particleFlowSuperClusterOOTECAL_*_*',
        'keep recoCaloClusters_particleFlowSuperClusterOOTECAL_*_*',
        'keep recoTracks_GsfGlobalElectronTest_*_*',
        'keep recoGsfTracks_electronGsfTracks_*_*',
        'keep recoTracks_generalTracks_*_*',
        'keep recoTracks_conversionStepTracks_*_*',
        'keep recoTracks_beamhaloTracks_*_*',
        'keep recoTracks_ctfPixelLess_*_*',
        'keep *_dedxHarmonic2_*_*',
        'keep *_dedxPixelHarmonic2_*_*',
        'keep *_dedxHitInfo_*_*',
        'keep *_trackExtrapolator_*_*',
        'keep *_generalTracks_MVAValues_*',
        'keep *_generalTracks_MVAVals_*',
        'keep recoCaloJets_ak4CaloJets_*_*',
        'keep *_ak4CaloJets_rho_*',
        'keep *_ak4CaloJets_sigma_*',
        'keep *_ak4PFJetsCHS_*_*',
        'keep floatedmValueMap_puppi_*_*',
        'keep *_ak4PFJetsPuppi_*_*',
        'keep *_ak8PFJetsPuppi_*_*',
        'keep *_ak8PFJetsPuppiSoftDrop_*_*',
        'keep recoPFJets_ak4PFJets_*_*',
        'keep *_ak4PFJets_rho_*',
        'keep *_ak4PFJets_sigma_*',
        'keep *_JetPlusTrackZSPCorJetAntiKt4_*_*',
        'keep *_caloTowers_*_*',
        'keep *_CastorTowerReco_*_*',
        'keep *_ak4JetTracksAssociatorAtVertex_*_*',
        'keep *_ak4JetTracksAssociatorAtVertexPF_*_*',
        'keep *_ak4JetTracksAssociatorExplicit_*_*',
        'keep *_ak4JetExtender_*_*',
        'keep *_ak4JetID_*_*',
        'keep recoBasicJets_ak5CastorJets_*_*',
        'keep *_ak5CastorJets_rho_*',
        'keep *_ak5CastorJets_sigma_*',
        'keep *_ak5CastorJetID_*_*',
        'keep recoBasicJets_ak7CastorJets_*_*',
        'keep *_ak7CastorJets_rho_*',
        'keep *_ak7CastorJets_sigma_*',
        'keep *_ak7CastorJetID_*_*',
        'keep *_fixedGridRhoAll_*_*',
        'keep *_fixedGridRhoFastjetAll_*_*',
        'keep *_fixedGridRhoFastjetAllTmp_*_*',
        'keep *_fixedGridRhoFastjetCentral_*_*',
        'keep *_fixedGridRhoFastjetAllCalo_*_*',
        'keep *_fixedGridRhoFastjetCentralCalo_*_*',
        'keep *_fixedGridRhoFastjetCentralChargedPileUp_*_*',
        'keep *_fixedGridRhoFastjetCentralNeutral_*_*',
        'keep *_ak8PFJetsPuppiSoftDropMass_*_*',
        'keep recoCaloMETs_caloMet_*_*',
        'keep recoCaloMETs_caloMetBE_*_*',
        'keep recoCaloMETs_caloMetBEFO_*_*',
        'keep recoCaloMETs_caloMetM_*_*',
        'keep recoPFMETs_pfMet_*_*',
        'keep recoPFMETs_pfChMet_*_*',
        'keep floatedmValueMap_puppiNoLep_*_*',
        'keep recoPFMETs_pfMetPuppi_*_*',
        'keep recoMuonMETCorrectionDataedmValueMap_muonMETValueMapProducer_*_*',
        'keep HcalNoiseSummary_hcalnoise_*_*',
        'keep recoGlobalHaloData_GlobalHaloData_*_*',
        'keep recoCSCHaloData_CSCHaloData_*_*',
        'keep recoBeamHaloSummary_BeamHaloSummary_*_*',
        'keep recoMuons_muons_*_*',
        'keep booledmValueMap_muons_*_*',
        'keep doubleedmValueMap_muons_muPFMean*_*',
        'keep doubleedmValueMap_muons_muPFSum*_*',
        'keep *_muons_muonShowerInformation_*',
        'keep recoMuonTimeExtraedmValueMap_muons_*_*',
        'keep recoMuonCosmicCompatibilityedmValueMap_muons_*_*',
        'keep uintedmValueMap_muons_*_*',
        'keep *_particleFlow_muons_*',
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
        'keep recoTracks_displacedGlobalMuons_*_*',
        'keep recoTrackExtras_displacedGlobalMuons_*_*',
        'keep TrackingRecHitsOwned_displacedGlobalMuons_*_*',
        'keep recoTracks_cosmicMuons_*_*',
        'keep recoMuons_muonsFromCosmics_*_*',
        'keep recoTracks_cosmicMuons1Leg_*_*',
        'keep recoMuons_muonsFromCosmics1Leg_*_*',
        'keep recoTracks_refittedStandAloneMuons_*_*',
        'keep recoTrackExtras_refittedStandAloneMuons_*_*',
        'keep TrackingRecHitsOwned_refittedStandAloneMuons_*_*',
        'keep recoTracks_displacedStandAloneMuons__*',
        'keep recoTrackExtras_displacedStandAloneMuons_*_*',
        'keep TrackingRecHitsOwned_displacedStandAloneMuons_*_*',
        'keep *_muonReducedTrackExtras_*_*',
        'keep *_softPFElectronBJetTags_*_*',
        'keep *_softPFMuonBJetTags_*_*',
        'keep *_pfTrackCountingHighEffBJetTags_*_*',
        'keep *_pfJetProbabilityBJetTags_*_*',
        'keep *_pfJetBProbabilityBJetTags_*_*',
        'keep *_pfSimpleSecondaryVertexHighEffBJetTags_*_*',
        'keep *_pfSimpleInclusiveSecondaryVertexHighEffBJetTags_*_*',
        'keep *_pfCombinedSecondaryVertexV2BJetTags_*_*',
        'keep *_pfCombinedInclusiveSecondaryVertexV2BJetTags_*_*',
        'keep *_pfGhostTrackBJetTags_*_*',
        'keep *_pfCombinedMVAV2BJetTags_*_*',
        'keep *_inclusiveCandidateSecondaryVertices_*_*',
        'keep *_inclusiveCandidateSecondaryVerticesCvsL_*_*',
        'keep *_pfCombinedCvsLJetTags_*_*',
        'keep *_pfCombinedCvsBJetTags_*_*',
        'keep *_pfChargeBJetTags_*_*',
        'keep *_pfDeepCSVJetTags_*_*',
        'keep *_pfDeepCMVAJetTags_*_*',
        'keep *_pixelClusterTagInfos_*_*',
        'keep recoRecoTauPiZeros_hpsPFTauProducer_pizeros_*',
        'keep recoPFTaus_hpsPFTauProducer_*_*',
        'keep *_hpsPFTauBasicDiscriminators_*_*',
        'keep *_hpsPFTauBasicDiscriminatorsdR03_*_*',
        'keep *_hpsPFTauDiscriminationByDeadECALElectronRejection_*_*',
        'keep *_hpsPFTauDiscriminationByDecayModeFinding_*_*',
        'keep *_hpsPFTauDiscriminationByDecayModeFindingNewDMs_*_*',
        'keep *_hpsPFTauDiscriminationByDecayModeFindingOldDMs_*_*',
        'keep *_hpsPFTauDiscriminationByMuonRejection3_*_*',
        'keep *_hpsPFTauTransverseImpactParameters_*_*',
        'keep  *_offlinePrimaryVertices__*',
        'keep *_offlinePrimaryVerticesWithBS_*_*',
        'keep *_offlinePrimaryVerticesFromCosmicTracks_*_*',
        'keep *_nuclearInteractionMaker_*_*',
        'keep *_generalV0Candidates_*_*',
        'keep *_inclusiveSecondaryVertices_*_*',
        'keep recoGsfElectronCores_gsfElectronCores_*_*',
        'keep recoGsfElectronCores_gedGsfElectronCores_*_*',
        'keep recoGsfElectrons_gsfElectrons_*_*',
        'keep recoGsfElectrons_gedGsfElectrons_*_*',
        'keep recoGsfElectronCores_uncleanedOnlyGsfElectronCores_*_*',
        'keep recoGsfElectrons_uncleanedOnlyGsfElectrons_*_*',
        'keep floatedmValueMap_eidRobustLoose_*_*',
        'keep floatedmValueMap_eidRobustTight_*_*',
        'keep floatedmValueMap_eidRobustHighEnergy_*_*',
        'keep floatedmValueMap_eidLoose_*_*',
        'keep floatedmValueMap_eidTight_*_*',
        'keep *_egmGedGsfElectronPFIsolation_*_*',
        'keep recoPhotonCores_gedPhotonCore_*_*',
        'keep recoPhotons_gedPhotons_*_*',
        'keep *_particleBasedIsolation_*_*',
        'keep recoPhotonCores_photonCore_*_*',
        'keep recoPhotons_photons_*_*',
        'keep recoPhotonCores_ootPhotonCore_*_*',
        'keep recoPhotons_ootPhotons_*_*',
        'keep recoConversions_conversions_*_*',
        'drop recoConversions_conversions_uncleanedConversions_*',
        'keep recoConversions_mustacheConversions_*_*',
        'keep *_gsfTracksOpenConversions_*_*',
        'keep recoConversions_allConversions_*_*',
        'keep recoConversions_allConversionsOldEG_*_*',
        'keep recoTracks_ckfOutInTracksFromConversions_*_*',
        'keep recoTracks_ckfInOutTracksFromConversions_*_*',
        'keep recoConversions_uncleanedOnlyAllConversions_*_*',
        'keep recoTracks_uncleanedOnlyCkfOutInTracksFromConversions_*_*',
        'keep recoTracks_uncleanedOnlyCkfInOutTracksFromConversions_*_*',
        'keep *_PhotonIDProd_*_*',
        'keep *_PhotonIDProdGED_*_*',
        'keep *_hfRecoEcalCandidate_*_*',
        'keep *_hfEMClusters_*_*',
        'keep *_gedGsfElectronCores_*_*',
        'keep *_gedGsfElectrons_*_*',
        'keep recoCaloClusters_lowPtGsfElectronSuperClusters_*_*',
        'keep recoGsfElectrons_lowPtGsfElectrons_*_*',
        'keep recoGsfElectronCores_lowPtGsfElectronCores_*_*',
        'keep recoGsfTracks_lowPtGsfEleGsfTracks_*_*',
        'keep *_lowPtGsfToTrackLinks_*_*',
        'keep recoSuperClusters_lowPtGsfElectronSuperClusters_*_*',
        'keep floatedmValueMap_lowPtGsfElectronSeedValueMaps_*_*',
        'keep floatedmValueMap_rekeyLowPtGsfElectronSeedValueMaps_*_*',
        'keep floatedmValueMap_lowPtGsfElectronID_*_*',
        'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*',
        'keep recoPFRecHits_particleFlowRecHitHBHE_Cleaned_*',
        'keep recoPFRecHits_particleFlowRecHitHF_Cleaned_*',
        'keep recoPFRecHits_particleFlowRecHitHO_Cleaned_*',
        'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*',
        'keep recoCaloClusters_particleFlowEGamma_*_*',
        'keep recoSuperClusters_particleFlowEGamma_*_*',
        'keep recoCaloClusters_particleFlowSuperClusterECAL_*_*',
        'keep recoSuperClusters_particleFlowSuperClusterECAL_*_*',
        'keep recoConversions_particleFlowEGamma_*_*',
        'keep recoPFCandidates_particleFlow_*_*',
        'keep recoPFCandidates_particleFlowTmp_AddedMuonsAndHadrons_*',
        'keep recoPFCandidates_particleFlowTmp_CleanedCosmicsMuons_*',
        'keep recoPFCandidates_particleFlowTmp_CleanedFakeMuons_*',
        'keep recoPFCandidates_particleFlowTmp_CleanedHF_*',
        'keep recoPFCandidates_particleFlowTmp_CleanedPunchThroughMuons_*',
        'keep recoPFCandidates_particleFlowTmp_CleanedPunchThroughNeutralHadrons_*',
        'keep recoPFCandidates_particleFlowTmp_CleanedTrackerAndGlobalMuons_*',
        'keep *_particleFlow_electrons_*',
        'keep *_particleFlow_photons_*',
        'keep *_particleFlow_muons_*',
        'keep recoCaloClusters_pfElectronTranslator_*_*',
        'keep recoPreshowerClusters_pfElectronTranslator_*_*',
        'keep recoSuperClusters_pfElectronTranslator_*_*',
        'keep recoCaloClusters_pfPhotonTranslator_*_*',
        'keep recoPreshowerClusters_pfPhotonTranslator_*_*',
        'keep recoSuperClusters_pfPhotonTranslator_*_*',
        'keep recoPhotons_pfPhotonTranslator_*_*',
        'keep recoPhotonCores_pfPhotonTranslator_*_*',
        'keep recoConversions_pfPhotonTranslator_*_*',
        'keep *_particleFlowPtrs_*_*',
        'keep *_particleFlowTmpPtrs_*_*',
        'keep *_chargedHadronPFTrackIsolation_*_*',
        'keep *_offlineBeamSpot_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_l1GtRecord_*_*',
        'keep *_l1GtTriggerMenuLite_*_*',
        'keep *_conditionsInEdm_*_*',
        'keep *_l1extraParticles_*_*',
        'keep *_l1L1GtObjectMap_*_*',
        'keep LumiSummary_lumiProducer_*_*',
        'drop *_hlt*_*_*',
        'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
        'keep edmTriggerResults_*_*_*',
        'keep triggerTriggerEvent_*_*_*',
        'keep *_hltFEDSelectorL1_*_*',
        'keep *_hltScoutingEgammaPacker_*_*',
        'keep *_hltScoutingMuonPacker_*_*',
        'keep *_hltScoutingPFPacker_*_*',
        'keep *_hltScoutingPrimaryVertexPacker_*_*',
        'keep *_hltScoutingTrackPacker_*_*',
        'keep edmTriggerResults_*_*_*',
        'keep L1AcceptBunchCrossings_scalersRawToDigi_*_*',
        'keep L1TriggerScalerss_scalersRawToDigi_*_*',
        'keep Level1TriggerScalerss_scalersRawToDigi_*_*',
        'keep LumiScalerss_scalersRawToDigi_*_*',
        'keep BeamSpotOnlines_scalersRawToDigi_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
        'keep CTPPSRecord_onlineMetaDataDigis_*_*',
        'keep DCSRecord_onlineMetaDataDigis_*_*',
        'keep OnlineLuminosityRecord_onlineMetaDataDigis_*_*',
        'keep recoBeamSpot_onlineMetaDataDigis_*_*',
        'keep *_tcdsDigis_*_*',
        'keep *_logErrorHarvester_*_*',
        'keep FEDRawDataCollection_rawDataCollector_*_*',
        'keep FEDRawDataCollection_source_*_*',
        'drop *_MEtoEDMConverter_*_*',
        'drop *_*_*_SKIM',
        'keep *_hbhereco__*'
     ) )
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '120X_mcRun3_2021_realistic_v6', '')

# Path and EndPath definitions
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)
process.SKIMStreamEXOLLPJetHCALOutPath = cms.EndPath(process.SKIMStreamEXOLLPJetHCAL)

# Schedule definition
process.schedule = cms.Schedule(process.EXOLLPJetHCALPath,process.RECOSIMoutput_step,process.SKIMStreamEXOLLPJetHCALOutPath)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)



# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
