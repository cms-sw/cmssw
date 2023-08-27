# Auto generated configuration file
# using:
# Revision: 1.222.2.1
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v
# with command line options: skim -s SKIM:LogError+DiJet --dbs find file,file.parent where dataset=/MinimumBias/Commissioning10-PromptReco-v7/RECO  and run.number=132440 -n 100 --python_file JetPDSkim_cfg.py --no_exec --data --magField AutoFromDBCurrent --scenario pp --conditions auto:com10
import FWCore.ParameterSet.Config as cms

process = cms.Process('SKIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.Skims_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.222.2.1 $'),
    annotation = cms.untracked.string('skim nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.options = cms.untracked.PSet(

)
# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring('/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/544F17A6-F53B-DF11-BD0A-000423D98E54.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/D83F0CD9-F43B-DF11-B950-00151796CD80.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/049F45A5-F53B-DF11-A9FC-0030487A3DE0.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/6656A751-F63B-DF11-B6BF-000423D94908.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/3EF76888-EE3B-DF11-8699-001D09F232B9.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/0E7455C3-1A3C-DF11-A11B-000423D99E46.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/8222CDEF-EF3B-DF11-887D-0019B9F7312C.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/1ED1A55A-F83B-DF11-A447-000423D98920.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/3802F8FA-F63B-DF11-A4A8-0030487A195C.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/140A72D3-ED3B-DF11-9C4C-000423D9970C.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/D0354B8A-EE3B-DF11-AE26-001D09F24600.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/2618D19B-0A3C-DF11-A660-0030487A1990.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/64E7C9A0-F03B-DF11-932D-000423D9890C.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/885AEF26-F43B-DF11-92B0-000423D985B0.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/B028B925-F43B-DF11-A5D8-000423D94E70.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/1630185A-F83B-DF11-BC24-000423D99160.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/8A8AC6B0-F73B-DF11-B237-000423D99658.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/4246B609-FA3B-DF11-AF4E-0030487CD6B4.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/EA896425-F93B-DF11-B971-000423D99AAE.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/08551EF0-EF3B-DF11-92E2-001D09F28F0C.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/3E945B39-EF3B-DF11-BD3D-000423D99996.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/26C30C72-F33B-DF11-9B78-000423D998BA.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/D67F8DC6-F23B-DF11-B98D-000423D8F63C.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/5ECBB822-ED3B-DF11-86C4-000423D99CEE.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/1CF54554-F13B-DF11-8BFB-000423D98BC4.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/CEF82055-F13B-DF11-BF11-000423D9989E.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/5C217F68-EC3B-DF11-8634-0030487CD7B4.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/16D1A60D-F23B-DF11-B86D-000423D99A8E.root',
        '/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/CEB753C6-F23B-DF11-9028-000423D99AAA.root'),
    fileNames = cms.untracked.vstring('/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/F4C92A98-163C-DF11-9788-0030487C7392.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/F4C92A98-163C-DF11-9788-0030487C7392.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/F427D642-173C-DF11-A909-0030487C60AE.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/F427D642-173C-DF11-A909-0030487C60AE.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/E27821C3-0C3C-DF11-9BD9-0030487CD718.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/D87D5469-2E3C-DF11-A470-000423D99896.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/B647CAD9-0E3C-DF11-886F-0030487CD716.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/A860D55E-193C-DF11-BE29-0030487C60AE.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/A860D55E-193C-DF11-BE29-0030487C60AE.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/9884BC11-0C3C-DF11-8F9C-000423D986C4.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/9884BC11-0C3C-DF11-8F9C-000423D986C4.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/92684831-233C-DF11-ABA0-0030487CD16E.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/90269E76-0D3C-DF11-A1A0-0030487CD840.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/8CAE3014-133C-DF11-A05D-000423D174FE.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/8CAE3014-133C-DF11-A05D-000423D174FE.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/8C51BAC6-1A3C-DF11-A0EE-000423D94A04.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/8C51BAC6-1A3C-DF11-A0EE-000423D94A04.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/8C042B04-2D3C-DF11-939F-0030487CD178.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/8C042B04-2D3C-DF11-939F-0030487CD178.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/80471A6B-0E3C-DF11-8DCD-0030487C6A66.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/762824C3-0C3C-DF11-A4FD-0030487CD6D2.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/6A3533F5-103C-DF11-B3AA-00304879BAB2.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/6A3533F5-103C-DF11-B3AA-00304879BAB2.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/4C8979D2-073C-DF11-B97B-000423D6AF24.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/26C8DED9-0E3C-DF11-9D83-0030487CD7B4.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/26C8DED9-0E3C-DF11-9D83-0030487CD7B4.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/181C44F7-093C-DF11-A9CB-001D09F24FEC.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/0AA7C390-0F3C-DF11-BD65-000423D998BA.root',
        '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/0AA7C390-0F3C-DF11-BD65-000423D998BA.root')
)

# Output definition

# Additional output definition
process.SKIMStreamDiJet = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('diJetAveSkimPath')
    ),
    outputCommands = cms.untracked.vstring('drop *',
        'keep recoCaloJets_kt4CaloJets_*_*',
        'keep recoCaloJets_kt6CaloJets_*_*',
        'keep recoCaloJets_ak4CaloJets_*_*',
        'keep recoCaloJets_ak7CaloJets_*_*',
        'keep recoCaloJets_iterativeCone5CaloJets_*_*',
        'keep *_kt4JetID_*_*',
        'keep *_kt6JetID_*_*',
        'keep *_ak4JetID_*_*',
        'keep *_ak7JetID_*_*',
        'keep *_ic5JetID_*_*',
        'keep recoPFJets_kt4PFJets_*_*',
        'keep recoPFJets_kt6PFJets_*_*',
        'keep recoPFJets_ak4PFJets_*_*',
        'keep recoPFJets_ak7PFJets_*_*',
        'keep recoPFJets_iterativeCone5PFJets_*_*',
        'keep *_JetPlusTrackZSPCorJetAntiKt5_*_*',
        'keep edmTriggerResults_TriggerResults_*_*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep L1GlobalTriggerObjectMapRecord_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_*_*_*',
        'keep recoTracks_generalTracks_*_*',
        'keep *_towerMaker_*_*',
        'keep *_EventAuxilary_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_hcalnoise_*_*',
        'keep *_metHO_*_*',
        'keep *_metNoHF_*_*',
        'keep *_metNoHFHO_*_*',
        'keep *_met_*_*'),
    fileName = cms.untracked.string('DiJet.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('DiJet'),
        dataTier = cms.untracked.string('USER')
    )
)
process.SKIMStreamLogError = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathlogerror')
    ),
    outputCommands = (cms.untracked.vstring('drop *', 'drop *', 'keep  FEDRawDataCollection_rawDataCollector_*_*', 'keep  FEDRawDataCollection_source_*_*', 'keep  FEDRawDataCollection_rawDataCollector_*_*', 'keep  FEDRawDataCollection_source_*_*', 'drop *_hlt*_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep FEDRawDataCollection_rawDataCollector_*_*', 'keep FEDRawDataCollection_source_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep DetIds_siStripDigis_*_*', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep HBHERecHitsSorted_hbherecoMB_*_*', 'keep HORecHitsSorted_horecoMB_*_*', 'keep HFRecHitsSorted_hfrecoMB_*_*', 'keep ZDCDataFramesSorted_*Digis_*_*', 'keep ZDCRecHitsSorted_*_*_*', 'keep *_castorreco_*_*', 'keep *_CastorTowerReco_*_*', 'keep *_CastorClusterRecoAntiKt07_*_*', 'keep *_CastorJetEgammaRecoAntiKt07_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep EBSrFlagsSorted_ecalDigis_*_*', 'keep EESrFlagsSorted_ecalDigis_*_*', 'keep *_selectDigi_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHits*_*_*', 'keep *_hybridSuperClusters_*_*', 'keep *_uncleanedHybridSuperClusters_*_*', 'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 'keep *_multi5x5BasicClusters_*_*', 'keep recoSuperClusters_multi5x5SuperClusters_*_*', 'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*', 'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*', 'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*', 'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*', 'drop recoClusterShapes_*_*_*', 'drop recoBasicClustersToOnerecoClusterShapesAssociation_*_*_*', 'drop recoBasicClusters_multi5x5BasicClusters_multi5x5BarrelBasicClusters_*', 'drop recoSuperClusters_multi5x5SuperClusters_multi5x5BarrelSuperClusters_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_electronMergedSeeds_*_*', 'keep recoGsfTracks_electronGsfTracks_*_*', 'keep recoGsfTrackExtras_electronGsfTracks_*_*', 'keep recoTrackExtras_electronGsfTracks_*_*', 'keep TrackingRecHitsOwned_electronGsfTracks_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoTrackExtras_generalTracks_*_*', 'keep TrackingRecHitsOwned_generalTracks_*_*', 'keep recoTracks_beamhaloTracks_*_*', 'keep recoTrackExtras_beamhaloTracks_*_*', 'keep TrackingRecHitsOwned_beamhaloTracks_*_*', 'keep recoTracks_regionalCosmicTracks_*_*', 'keep recoTrackExtras_regionalCosmicTracks_*_*', 'keep TrackingRecHitsOwned_regionalCosmicTracks_*_*', 'keep recoTracks_rsWithMaterialTracks_*_*', 'keep recoTrackExtras_rsWithMaterialTracks_*_*', 'keep TrackingRecHitsOwned_rsWithMaterialTracks_*_*', 'keep *_ctfPixelLess_*_*', 'keep *_dedxTruncated40_*_*', 'keep *_dedxMedian_*_*', 'keep *_dedxHarmonic2_*_*', 'keep *_kt4CaloJets_*_*', 'keep *_kt6CaloJets_*_*', 'keep *_ak4CaloJets_*_*', 'keep *_ak7CaloJets_*_*', 'keep *_iterativeCone5CaloJets_*_*', 'keep *_iterativeCone15CaloJets_*_*', 'keep *_kt4PFJets_*_*', 'keep *_kt6PFJets_*_*', 'keep *_ak4PFJets_*_*', 'keep *_ak7PFJets_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_JetPlusTrackZSPCorJetAntiKt5_*_*', 'keep *_ak4TrackJets_*_*', 'keep *_kt4TrackJets_*_*', 'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep *_ak4JetTracksAssociatorAtVertex_*_*', 'keep *_ak4JetTracksAssociatorAtCaloFace_*_*', 'keep *_ak4JetExtender_*_*', 'keep *_ak7JetTracksAssociatorAtVertex_*_*', 'keep *_ak7JetTracksAssociatorAtCaloFace_*_*', 'keep *_ak7JetExtender_*_*', 'keep *_ak4JetID_*_*', 'keep *_ak7JetID_*_*', 'keep *_ic5JetID_*_*', 'keep *_kt4JetID_*_*', 'keep *_kt6JetID_*_*', 'keep *_trackExtrapolator_*_*', 'keep recoCaloMETs_*_*_*', 'keep recoMETs_*_*_*', 'keep recoPFMETs_*_*_*', 'keep recoMuonMETCorrectionDataedmValueMap_*_*_*', 'keep recoHcalNoiseRBXs_*_*_*', 'keep HcalNoiseSummary_*_*_*', 'keep *HaloData_*_*_*', 'keep *BeamHaloSummary_*_*_*', 'keep *_MuonSeed_*_*', 'keep *_ancientMuonSeed_*_*', 'keep *_mergedStandAloneMuonSeeds_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep TrackingRecHitsOwned_tevMuons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_CosmicMuonSeed_*_*', 'keep recoTrackExtras_cosmicMuons_*_*', 'keep TrackingRecHitsOwned_cosmicMuons_*_*', 'keep recoTrackExtras_globalCosmicMuons_*_*', 'keep TrackingRecHitsOwned_globalCosmicMuons_*_*', 'keep recoTrackExtras_cosmicMuons1Leg_*_*', 'keep TrackingRecHitsOwned_cosmicMuons1Leg_*_*', 'keep recoTrackExtras_globalCosmicMuons1Leg_*_*', 'keep TrackingRecHitsOwned_globalCosmicMuons1Leg_*_*', 'keep recoTracks_cosmicsVetoTracks_*_*', 'keep *_SETMuonSeed_*_*', 'keep recoTracks_standAloneSETMuons_*_*', 'keep recoTrackExtras_standAloneSETMuons_*_*', 'keep TrackingRecHitsOwned_standAloneSETMuons_*_*', 'keep recoTracks_globalSETMuons_*_*', 'keep recoTrackExtras_globalSETMuons_*_*', 'keep TrackingRecHitsOwned_globalSETMuons_*_*', 'keep recoMuons_muonsWithSET_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep recoTracks_tevMuons_*_*', 'keep recoTrackExtras_tevMuons_*_*', 'keep recoTracksToOnerecoTracksAssociation_tevMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoMuons_muons_*_*', 'keep booledmValueMap_muid*_*_*', 'keep recoMuonTimeExtraedmValueMap_muons_*_*', 'keep recoTracks_cosmicMuons_*_*', 'keep recoTracks_globalCosmicMuons_*_*', 'keep recoMuons_muonsFromCosmics_*_*', 'keep recoTracks_cosmicMuons1Leg_*_*', 'keep recoTracks_globalCosmicMuons1Leg_*_*', 'keep recoMuons_muonsFromCosmics1Leg_*_*', 'keep recoMuonCosmicCompatibilityedmValueMap_cosmicsVeto_*_*', 'keep uintedmValueMap_cosmicsVeto_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_ghostTrackVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_simpleSecondaryVertexHighEffBJetTags_*_*', 'keep *_simpleSecondaryVertexHighPurBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_ghostTrackBJetTags_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronCands_*_*', 'keep *_softPFElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softElectronByIP3dBJetTags_*_*', 'keep *_softElectronByPtBJetTags_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonByIP3dBJetTags_*_*', 'keep *_softMuonByPtBJetTags_*_*', 'keep *_combinedMVABJetTags_*_*', 'keep *_pfRecoTauTagInfoProducer_*_*', 'keep *_fixedConePFTauProducer*_*_*', 'keep *_fixedConePFTauDiscrimination*_*_*', 'keep *_hpsPFTauProducer*_*_*', 'keep *_hpsPFTauDiscrimination*_*_*', 'keep *_shrinkingConePFTauProducer*_*_*', 'keep *_shrinkingConePFTauDecayModeIndexProducer*_*_*', 'keep *_shrinkingConePFTauDiscrimination*_*_*', 'keep *_TCTauJetPlusTrackZSPCorJetAntiKt5_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep recoCaloTaus_caloRecoTauProducer*_*_*', 'keep *_caloRecoTauDiscrimination*_*_*', 'keep  *_offlinePrimaryVertices_*_*', 'keep  *_offlinePrimaryVerticesWithBS_*_*', 'keep  *_offlinePrimaryVerticesFromCosmicTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep *_generalV0Candidates_*_*', 'keep recoGsfElectronCores_gsfElectronCores_*_*', 'keep recoGsfElectrons_gedGsfElectrons_*_*', 'keep floatedmValueMap_eidRobustLoose_*_*', 'keep floatedmValueMap_eidRobustTight_*_*', 'keep floatedmValueMap_eidRobustHighEnergy_*_*', 'keep floatedmValueMap_eidLoose_*_*', 'keep floatedmValueMap_eidTight_*_*', 'keep recoPhotons_photons_*_*', 'keep recoPhotonCores_photonCore_*_*', 'keep recoConversions_conversions_*_*', 'drop *_conversions_uncleanedConversions_*', 'keep recoConversions_trackerOnlyConversions_*_*', 'keep recoTracks_ckfOutInTracksFromConversions_*_*', 'keep recoTracks_ckfInOutTracksFromConversions_*_*', 'keep recoTrackExtras_ckfOutInTracksFromConversions_*_*', 'keep recoTrackExtras_ckfInOutTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfOutInTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfInOutTracksFromConversions_*_*', 'keep *_PhotonIDProd_*_*', 'keep *_hfRecoEcalCandidate_*_*', 'keep *_hfEMClusters_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFRecHits_*_Cleaned_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep recoPFDisplacedVertexs_*_*_*', 'keep *_pfElectronTranslator_*_*', 'keep *_trackerDrivenElectronSeeds_preid_*', 'keep *_offlineBeamSpot_*_*', 'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtTriggerMenuLite_*_*', 'keep *_conditionsInEdm_*_*', 'keep *_l1extraParticles_*_*', 'keep L1MuGMTReadoutCollection_gtDigis_*_*', 'keep L1GctEmCand*_gctDigis_*_*', 'keep L1GctJetCand*_gctDigis_*_*', 'keep L1GctEtHad*_gctDigis_*_*', 'keep L1GctEtMiss*_gctDigis_*_*', 'keep L1GctEtTotal*_gctDigis_*_*')+cms.untracked.vstring('keep L1GctHtMiss*_gctDigis_*_*', 'keep L1GctJetCounts*_gctDigis_*_*', 'keep L1GctHFRingEtSums*_gctDigis_*_*', 'keep L1GctHFBitCounts*_gctDigis_*_*', 'keep LumiDetails_lumiProducer_*_*', 'keep LumiSummary_lumiProducer_*_*', 'drop *_hlt*_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep *_MEtoEDMConverter_*_*', 'keep L1AcceptBunchCrossings_*_*_*', 'keep L1TriggerScalerss_*_*_*', 'keep Level1TriggerScalerss_*_*_*', 'keep LumiScalerss_*_*_*', 'keep BeamSpotOnlines_*_*_*', 'keep DcsStatuss_*_*_*', 'keep *_logErrorHarvester_*_*', 'drop *_MEtoEDMConverter_*_*', 'drop *_*_*_SKIM')),
    fileName = cms.untracked.string('LogError.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('LogError'),
        dataTier = cms.untracked.string('RAW-RECO')
    )
)

import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
process.HTSD = hlt.triggerResultsFilter.clone(
    triggerConditions = cms.vstring('HLT_HT*',),
    hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
    l1tResults = cms.InputTag(""),
    throw = cms.bool( False )
    )
process.pathHTSDSkimPath = cms.Path( process.HTSD )

process.SKIMStreamHTSD = process.SKIMStreamLogError.clone(
    fileName = cms.untracked.string('HTSD.root'),
     SelectEvents = cms.untracked.PSet(
     SelectEvents = cms.vstring('pathHTSDSkimPath')
    ),
    dataset = cms.untracked.PSet(
     filterName = cms.untracked.string('HTSD'),
     dataTier = cms.untracked.string('RAW-RECO')
    )
    )
process.SKIMStreamHTSDOutPath = cms.EndPath( process.SKIMStreamHTSD )


# Other statements
process.GlobalTag.globaltag = 'GR_R_38X_V13::All'

# Path and EndPath definitions
process.singlePhotonPt5SkimPath = cms.Path(process.singlePhotonPt5QualitySeq)
process.WpfMetSkimPath = cms.Path(process.pfMetWMuNuSeq)
process.pathrpcTecSkim = cms.Path(process.rpcTecSkimseq)
process.diJetAveSkimPath = cms.Path(process.DiJetAveSkim_Trigger)
process.pathpfgskim3noncross = cms.Path(process.pfgskim3noncrossseq)
process.pathdtSkim = cms.Path(process.dtSkimseq)
process.pathL1MuBitSkim = cms.Path(process.l1MuBitsSkimseq)
process.oniaSkimPath = cms.Path(process.oniaSkimSequence)
process.cosmicSPSkimPath = cms.Path(process.cosmicSPSkim)
process.pathCSCAloneSkim = cms.Path(process.cscSkimAloneSeq)
process.muonJPsiMMSkimPath = cms.Path(process.muonJPsiMMRecoQualitySeq)
process.pathCSCHLTSkim = cms.Path(process.cscHLTSkimSeq)
process.pathlogerror = cms.Path(process.logerrorseq)
process.jetSkimPath = cms.Path(process.jetRecoQualitySeq)
process.pathgoodcolhf = cms.Path(process.goodcollHFrequirement)
process.pathCSCSkim = cms.Path(process.cscSkimseq)
process.WtcMetSkimPath = cms.Path(process.tcMetWMuNuSeq)
process.tauSkimPath = cms.Path(process.tauSkimSequence)
process.goodvertexSkimPath = cms.Path(process.goodvertexSkim)
process.HSCPSkimPath = cms.Path(process.HSCPSkim)
process.singleElectronPt5SkimPath = cms.Path(process.singleElectronPt5RecoQualitySeq)
process.relvaltrackSkimPath = cms.Path(process.relvaltrackSkim)
process.pathgoodcoll1 = cms.Path(process.goodcollL1requirement)
process.WZEGSkimPath = cms.Path(process.WZfilterSkim)
process.singlePfTauPt15SkimPath = cms.Path(process.singlePfTauPt15QualitySeq)
process.relvalmuonSkimPath = cms.Path(process.relvalmuonSkim)
process.ZMuSkimPath = cms.Path(process.diMuonSelSeq)
process.muonTracksSkimPath = cms.Path(process.muonTracksSkim)
process.pathHLTdtSkim = cms.Path(process.dtHLTSkimseq)
process.ecalrechitSkimPath = cms.Path(process.ecalrechitSkim)
process.singleMuPt5SkimPath = cms.Path(process.singleMuPt5RecoQualitySeq)
process.SKIMStreamDiJetOutPath = cms.EndPath(process.SKIMStreamDiJet)
process.SKIMStreamLogErrorOutPath = cms.EndPath(process.SKIMStreamLogError)

# Schedule definition
process.schedule = cms.Schedule(process.pathlogerror,process.diJetAveSkimPath,process.pathHTSDSkimPath,process.SKIMStreamDiJetOutPath,process.SKIMStreamLogErrorOutPath,process.SKIMStreamHTSDOutPath)
