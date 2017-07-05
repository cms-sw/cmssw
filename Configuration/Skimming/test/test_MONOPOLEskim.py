# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: RECO -s RAW2DIGI,L1Reco,RECO,SKIM:EXOMONOPOLE,EI,PAT --runUnscheduled --nThreads 4 --data --era Run2_2016 --scenario pp --conditions 80X_dataRun2_2016LegacyRepro_v3 --eventcontent RECO --datatier RECO --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016,RecoTracker/Configuration/customizeMinPtForHitRecoveryInGluedDet.customizeHitRecoveryInGluedDetTkSeedsOnly --filein /store/data/Run2016E/SinglePhoton/RAW/v2/000/277/069/00000/0696A25D-0A4E-E611-99E9-02163E013842.root -n 100 --python_filename=test_MONOPOLEskim.py --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('Configuration.StandardSequences.Skims_cff')
process.load('CommonTools.ParticleFlow.EITopPAG_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Run2016E/SinglePhoton/RAW/v2/000/277/069/00000/0696A25D-0A4E-E611-99E9-02163E013842.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('RECO nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('RECO_RAW2DIGI_L1Reco_RECO_SKIM_EI_PAT.root'),
    outputCommands = process.RECOEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
process.SKIMStreamEXOMONOPOLE = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('EXOMONOPOLEPath')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('AOD'),
        filterName = cms.untracked.string('EXOMONOPOLE')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('EXOMONOPOLE.root'),
    outputCommands = cms.untracked.vstring( ('drop *', 
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
        'keep recoTracks_GsfGlobalElectronTest_*_*', 
        'keep recoGsfTracks_electronGsfTracks_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_conversionStepTracks_*_*', 
        'keep recoTracks_beamhaloTracks_*_*', 
        'keep recoTracks_ctfPixelLess_*_*', 
        'keep *_dedxHarmonic2_*_*', 
        'keep *_dedxHitInfo_*_*', 
        'keep *_trackExtrapolator_*_*', 
        'keep *_generalTracks_MVAValues_*', 
        'keep *_generalTracks_MVAVals_*', 
        'keep *_ak4CaloJets_*_*', 
        'keep *_ak4PFJetsCHS_*_*', 
        'keep *_ak8PFJetsCHS_*_*', 
        'keep *_ak8PFJetsCHSSoftDrop_*_*', 
        'keep *_cmsTopTagPFJetsCHS_*_*', 
        'keep *_ak4PFJets_*_*', 
        'keep *_JetPlusTrackZSPCorJetAntiKt4_*_*', 
        'keep *_ak4TrackJets_*_*', 
        'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*', 
        'keep *_caloTowers_*_*', 
        'keep *_CastorTowerReco_*_*', 
        'keep *_ak4JetTracksAssociatorAtVertex_*_*', 
        'keep *_ak4JetTracksAssociatorAtVertexPF_*_*', 
        'keep *_ak4JetTracksAssociatorExplicit_*_*', 
        'keep *_ak4JetExtender_*_*', 
        'keep *_ak4JetID_*_*', 
        'keep *_ak5CastorJets_*_*', 
        'keep *_ak5CastorJetID_*_*', 
        'keep *_ak7CastorJets_*_*', 
        'keep *_ak7CastorJetID_*_*', 
        'keep *_fixedGridRhoAll_*_*', 
        'keep *_fixedGridRhoFastjetAll_*_*', 
        'keep *_fixedGridRhoFastjetAllTmp_*_*', 
        'keep *_fixedGridRhoFastjetCentral_*_*', 
        'keep *_fixedGridRhoFastjetAllCalo_*_*', 
        'keep *_fixedGridRhoFastjetCentralCalo_*_*', 
        'keep *_fixedGridRhoFastjetCentralChargedPileUp_*_*', 
        'keep *_fixedGridRhoFastjetCentralNeutral_*_*', 
        'drop doubles_*Jets_rhos_*', 
        'drop doubles_*Jets_sigmas_*', 
        'keep *_ak8PFJetsCHSSoftDropMass_*_*', 
        'keep recoCaloMETs_caloMet_*_*', 
        'keep recoCaloMETs_caloMetBE_*_*', 
        'keep recoCaloMETs_caloMetBEFO_*_*', 
        'keep recoCaloMETs_caloMetM_*_*', 
        'keep recoPFMETs_pfMet_*_*', 
        'keep recoPFMETs_pfChMet_*_*', 
        'keep recoPFMETs_pfMetEI_*_*', 
        'keep recoMuonMETCorrectionDataedmValueMap_muonMETValueMapProducer_*_*', 
        'drop recoHcalNoiseRBXs_*_*_*', 
        'keep HcalNoiseSummary_hcalnoise_*_*', 
        'keep recoGlobalHaloData_GlobalHaloData_*_*', 
        'keep recoCSCHaloData_CSCHaloData_*_*', 
        'keep recoBeamHaloSummary_BeamHaloSummary_*_*', 
        'keep *_muons_*_*', 
        'keep *_particleFlow_muons_*', 
        'drop *_muons_muons1stStep2muonsMap_*', 
        'drop recoIsoDepositedmValueMap_muons_*_*', 
        'drop doubleedmValueMap_muons_muPFIso*_*', 
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
        'keep *_softPFElectronBJetTags_*_*', 
        'keep *_softPFMuonBJetTags_*_*', 
        'keep *_pfTrackCountingHighEffBJetTags_*_*', 
        'keep *_pfJetProbabilityBJetTags_*_*', 
        'keep *_pfJetBProbabilityBJetTags_*_*', 
        'keep *_pfSimpleSecondaryVertexHighEffBJetTags_*_*', 
        'keep *_pfSimpleInclusiveSecondaryVertexHighEffBJetTags_*_*', 
        'keep *_pfCombinedSecondaryVertexV2BJetTags_*_*', 
        'keep *_pfCombinedInclusiveSecondaryVertexV2BJetTags_*_*', 
        'keep *_pfCombinedMVAV2BJetTags_*_*', 
        'keep *_inclusiveCandidateSecondaryVertices_*_*', 
        'keep *_inclusiveCandidateSecondaryVerticesCvsL_*_*', 
        'keep *_pfCombinedCvsLJetTags_*_*', 
        'keep *_pfCombinedCvsBJetTags_*_*', 
        'keep recoRecoTauPiZeros_hpsPFTauProducer_pizeros_*', 
        'keep recoPFTaus_hpsPFTauProducer_*_*', 
        'keep *_hpsPFTauDiscriminationByDeadECALElectronRejection_*_*', 
        'keep *_hpsPFTauDiscriminationByDecayModeFinding_*_*', 
        'keep *_hpsPFTauDiscriminationByDecayModeFindingNewDMs_*_*', 
        'keep *_hpsPFTauDiscriminationByDecayModeFindingOldDMs_*_*', 
        'keep *_hpsPFTauDiscriminationByLooseIsolation_*_*', 
        'keep *_hpsPFTauDiscriminationByLooseChargedIsolation_*_*', 
        'keep *_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits_*_*', 
        'keep *_hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR03_*_*', 
        'keep *_hpsPFTauDiscriminationByLooseMuonRejection3_*_*', 
        'keep *_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits_*_*', 
        'keep *_hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR03_*_*', 
        'keep *_hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits_*_*', 
        'keep *_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits_*_*', 
        'keep *_hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR03_*_*', 
        'keep *_hpsPFTauDiscriminationByTightMuonRejection3_*_*', 
        'keep *_hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone_*_*', 
        'keep *_hpsPFTauNeutralIsoPtSum_*_*', 
        'keep *_hpsPFTauPUcorrPtSum_*_*', 
        'keep *_hpsPFTauChargedIsoPtSum_*_*', 
        'keep *_hpsPFTauTransverseImpactParameters_*_*', 
        'keep *_hpsPFTauFootprintCorrection_*_*', 
        'keep *_hpsPFTauNeutralIsoPtSumWeight_*_*', 
        'keep *_hpsPFTauPhotonPtSumOutsideSignalCone_*_*', 
        'keep *_hpsPFTauDiscriminationByMVA6rawElectronRejection_*_*', 
        'keep *_hpsPFTauDiscriminationByMVA6VLooseElectronRejection_*_*', 
        'keep *_hpsPFTauDiscriminationByMVA6LooseElectronRejection_*_*', 
        'keep *_hpsPFTauDiscriminationByMVA6MediumElectronRejection_*_*', 
        'keep *_hpsPFTauDiscriminationByMVA6TightElectronRejection_*_*', 
        'keep *_hpsPFTauDiscriminationByMVA6VTightElectronRejection_*_*', 
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw_*_*', 
        'keep *_hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw_*_*', 
        'keep *_hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw_*_*', 
        'keep *_hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByTightIsolationMVArun2v1PWoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWoldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTraw_*_*', 
        'keep *_hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWnewDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWnewDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByTightIsolationMVArun2v1PWnewDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWnewDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWnewDMwLT_*_*', 
        'keep *_hpsPFTauChargedIsoPtSumdR03_*_*', 
        'keep *_hpsPFTauNeutralIsoPtSumdR03_*_*', 
        'keep *_hpsPFTauNeutralIsoPtSumWeightdR03_*_*', 
        'keep *_hpsPFTauFootprintCorrectiondR03_*_*', 
        'keep *_hpsPFTauPhotonPtSumOutsideSignalConedR03_*_*', 
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw_*_*', 
        'keep *_hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBdR03oldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBdR03oldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByTightIsolationMVArun2v1DBdR03oldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBdR03oldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBdR03oldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw_*_*', 
        'keep *_hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWdR03oldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWdR03oldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByTightIsolationMVArun2v1PWdR03oldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWdR03oldDMwLT_*_*', 
        'keep *_hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWdR03oldDMwLT_*_*', 
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
        'keep *_photonEcalPFClusterIsolationProducer_*_*', 
        'keep *_electronEcalPFClusterIsolationProducer_*_*', 
        'keep *_photonHcalPFClusterIsolationProducer_*_*', 
        'keep *_electronHcalPFClusterIsolationProducer_*_*', 
        'drop *_egmGsfElectronIDs_*_*', 
        'drop *_egmPhotonIDs_*_*', 
        'keep recoPhotonCores_gedPhotonCore_*_*', 
        'keep recoPhotons_gedPhotons_*_*', 
        'keep *_particleBasedIsolation_*_*', 
        'drop *_gedPhotons_valMapPFEgammaCandToPhoton_*', 
        'keep recoPhotonCores_photonCore_*_*', 
        'keep recoPhotons_photons_*_*', 
        'keep recoConversions_conversions_*_*', 
        'keep recoConversions_mustacheConversions_*_*', 
        'drop *_conversions_uncleanedConversions_*', 
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
        'keep recoRecoEcalCandidates_hfRecoEcalCandidate_*_*', 
        'keep *_hfEMClusters_*_*', 
        'keep *_gedGsfElectronCores_*_*', 
        'keep *_gedGsfElectrons_*_*', 
        'drop CaloTowersSorted_towerMakerPF_*_*', 
        'drop *_pfElectronTranslator_*_*', 
        'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*', 
        'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*', 
        'keep recoPFRecHits_particleFlowClusterHO_Cleaned_*', 
        'keep recoPFRecHits_particleFlowClusterHF_Cleaned_*', 
        'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*', 
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
        'keep recoPFCandidates_particleFlowTmp_*_*', 
        'drop recoPFCandidates_particleFlowTmp__*', 
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
        'keep *_offlineBeamSpot_*_*', 
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtTriggerMenuLite_*_*', 
        'keep *_conditionsInEdm_*_*', 
        'keep *_l1extraParticles_*_*', 
        'keep *_l1L1GtObjectMap_*_*', 
        'keep LumiSummary_lumiProducer_*_*', 
        'keep *_gtStage2Digis__*', 
        'keep *_gmtStage2Digis_Muon_*', 
        'keep *_caloStage2Digis_Jet_*', 
        'keep *_caloStage2Digis_Tau_*', 
        'keep *_caloStage2Digis_EGamma_*', 
        'keep *_caloStage2Digis_EtSum_*', 
        'drop *_hlt*_*_*', 
        'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep triggerTriggerEvent_*_*_*', 
        'keep L1AcceptBunchCrossings_scalersRawToDigi_*_*', 
        'keep L1TriggerScalerss_scalersRawToDigi_*_*', 
        'keep Level1TriggerScalerss_scalersRawToDigi_*_*', 
        'keep LumiScalerss_scalersRawToDigi_*_*', 
        'keep BeamSpotOnlines_scalersRawToDigi_*_*', 
        'keep DcsStatuss_scalersRawToDigi_*_*', 
        'keep *_tcdsDigis_*_*', 
        'keep *_logErrorHarvester_*_*', 
        'keep *_pfIsolatedElectronsEI_*_*', 
        'keep *_pfIsolatedMuonsEI_*_*', 
        'keep recoPFJets_pfJetsEI_*_*', 
        'keep *_pfCombinedInclusiveSecondaryVertexV2BJetTagsEI_*_*', 
        'keep recoPFTaus_pfTausEI_*_*', 
        'keep recoPFTauDiscriminator_pfTausDiscriminationByDecayModeFinding_*_*', 
        'keep recoPFTauDiscriminator_pfTausDiscriminationByIsolation_*_*', 
        'keep recoPFMETs_pfMetEI_*_*', 
        'keep TotemTriggerCounters_totemTriggerRawToDigi_*_*', 
        'keep TotemFEDInfos_totemRPRawToDigi_*_*', 
        'keep TotemRPDigiedmDetSetVector_totemRPRawToDigi_*_*', 
        'keep TotemVFATStatusedmDetSetVector_totemRPRawToDigi_*_*', 
        'keep TotemRPClusteredmDetSetVector_totemRPClusterProducer_*_*', 
        'keep TotemRPRecHitedmDetSetVector_totemRPRecHitProducer_*_*', 
        'keep TotemRPUVPatternedmDetSetVector_totemRPUVPatternFinder_*_*', 
        'keep TotemRPLocalTrackedmDetSetVector_totemRPLocalTrackFitter_*_*', 
        'keep TotemFEDInfos_ctppsDiamondRawToDigi_*_*', 
        'keep CTPPSDiamondDigiedmDetSetVector_ctppsDiamondRawToDigi_*_*', 
        'keep TotemVFATStatusedmDetSetVector_ctppsDiamondRawToDigi_*_*', 
        'keep CTPPSDiamondRecHitedmDetSetVector_ctppsDiamondRecHits_*_*', 
        'keep CTPPSDiamondLocalTrackedmDetSetVector_ctppsDiamondLocalTracks_*_*', 
        'keep CTPPSLocalTrackLites_ctppsLocalTrackLiteProducer_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'keep *_multi5x5SuperClusters_multi5x5EndcapSuperClusters_*', 
        'keep *_multi5x5SuperClusters_uncleanOnlyMulti5x5EndcapBasicClusters_*', 
        'keep *_multi5x5SuperClusters_uncleanOnlyMulti5x5EndcapSuperClusters_*', 
        'keep *_siStripClusters_*_*', 
        'keep *_siPixelClusters_*_*', 
        'drop *_generalTracks_*_*', 
        'keep *_generalTracks_*_*', 
        'drop *_generalTracks_QualityMasks_*', 
        'keep *_ecalRecHit_EcalRecHitsEB_*', 
        'keep *_ecalRecHit_EcalRecHitsEE_*', 
        'keep *_hbhereco_*_*' ) )
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '80X_dataRun2_2016LegacyRepro_v3', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.eventinterpretaion_step = cms.Path(process.EIsequence)
process.Flag_trackingFailureFilter = cms.Path(process.goodVertices+process.trackingFailureFilter)
process.Flag_goodVertices = cms.Path(process.primaryVertexFilter)
process.Flag_CSCTightHaloFilter = cms.Path(process.CSCTightHaloFilter)
process.Flag_trkPOGFilters = cms.Path(process.trkPOGFilters)
process.Flag_HcalStripHaloFilter = cms.Path(process.HcalStripHaloFilter)
process.Flag_trkPOG_logErrorTooManyClusters = cms.Path(~process.logErrorTooManyClusters)
process.Flag_EcalDeadCellTriggerPrimitiveFilter = cms.Path(process.EcalDeadCellTriggerPrimitiveFilter)
process.Flag_ecalLaserCorrFilter = cms.Path(process.ecalLaserCorrFilter)
process.Flag_globalSuperTightHalo2016Filter = cms.Path(process.globalSuperTightHalo2016Filter)
process.Flag_eeBadScFilter = cms.Path(process.eeBadScFilter)
process.Flag_METFilters = cms.Path(process.metFilters)
process.Flag_chargedHadronTrackResolutionFilter = cms.Path(process.chargedHadronTrackResolutionFilter)
process.Flag_globalTightHalo2016Filter = cms.Path(process.globalTightHalo2016Filter)
process.Flag_CSCTightHaloTrkMuUnvetoFilter = cms.Path(process.CSCTightHaloTrkMuUnvetoFilter)
process.Flag_HBHENoiseIsoFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseIsoFilter)
process.Flag_BadChargedCandidateSummer16Filter = cms.Path(process.BadChargedCandidateSummer16Filter)
process.Flag_hcalLaserEventFilter = cms.Path(process.hcalLaserEventFilter)
process.Flag_BadPFMuonFilter = cms.Path(process.BadPFMuonFilter)
process.Flag_HBHENoiseFilter = cms.Path(process.HBHENoiseFilterResultProducer+process.HBHENoiseFilter)
process.Flag_trkPOG_toomanystripclus53X = cms.Path(~process.toomanystripclus53X)
process.Flag_EcalDeadCellBoundaryEnergyFilter = cms.Path(process.EcalDeadCellBoundaryEnergyFilter)
process.Flag_BadChargedCandidateFilter = cms.Path(process.BadChargedCandidateFilter)
process.Flag_trkPOG_manystripclus53X = cms.Path(~process.manystripclus53X)
process.Flag_BadPFMuonSummer16Filter = cms.Path(process.BadPFMuonSummer16Filter)
process.Flag_muonBadTrackFilter = cms.Path(process.muonBadTrackFilter)
process.Flag_CSCTightHalo2015Filter = cms.Path(process.CSCTightHalo2015Filter)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)
process.SKIMStreamEXOMONOPOLEOutPath = cms.EndPath(process.SKIMStreamEXOMONOPOLE)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.EXOMONOPOLEPath,process.eventinterpretaion_step,process.Flag_HBHENoiseFilter,process.Flag_HBHENoiseIsoFilter,process.Flag_CSCTightHaloFilter,process.Flag_CSCTightHaloTrkMuUnvetoFilter,process.Flag_CSCTightHalo2015Filter,process.Flag_globalTightHalo2016Filter,process.Flag_globalSuperTightHalo2016Filter,process.Flag_HcalStripHaloFilter,process.Flag_hcalLaserEventFilter,process.Flag_EcalDeadCellTriggerPrimitiveFilter,process.Flag_EcalDeadCellBoundaryEnergyFilter,process.Flag_goodVertices,process.Flag_eeBadScFilter,process.Flag_ecalLaserCorrFilter,process.Flag_trkPOGFilters,process.Flag_chargedHadronTrackResolutionFilter,process.Flag_muonBadTrackFilter,process.Flag_BadChargedCandidateFilter,process.Flag_BadPFMuonFilter,process.Flag_BadChargedCandidateSummer16Filter,process.Flag_BadPFMuonSummer16Filter,process.Flag_trkPOG_manystripclus53X,process.Flag_trkPOG_toomanystripclus53X,process.Flag_trkPOG_logErrorTooManyClusters,process.Flag_METFilters,process.endjob_step,process.RECOoutput_step,process.SKIMStreamEXOMONOPOLEOutPath)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(4)
process.options.numberOfStreams=cms.untracked.uint32(0)

# customisation of the process.

# Automatic addition of the customisation function from Configuration.DataProcessing.RecoTLR
from Configuration.DataProcessing.RecoTLR import customisePostEra_Run2_2016 

#call to customisation function customisePostEra_Run2_2016 imported from Configuration.DataProcessing.RecoTLR
process = customisePostEra_Run2_2016(process)

# Automatic addition of the customisation function from RecoTracker.Configuration.customizeMinPtForHitRecoveryInGluedDet
from RecoTracker.Configuration.customizeMinPtForHitRecoveryInGluedDet import customizeHitRecoveryInGluedDetTkSeedsOnly 

#call to customisation function customizeHitRecoveryInGluedDetTkSeedsOnly imported from RecoTracker.Configuration.customizeMinPtForHitRecoveryInGluedDet
process = customizeHitRecoveryInGluedDetTkSeedsOnly(process)

# End of customisation functions
#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)
process.load('Configuration.StandardSequences.PAT_cff')
from FWCore.ParameterSet.Utilities import cleanUnscheduled
process=cleanUnscheduled(process)

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllData 

#call to customisation function miniAOD_customizeAllData imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllData(process)

# End of customisation functions
