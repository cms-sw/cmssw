# Auto generated configuration file
# using: 
# Revision: 1.372.2.1 
# Source: /local/reps/CMSSW.admin/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: skims -s SKIM:@HT --data --no_exec --dbs find file,file.parent where dataset=/HT/Run2012A-PromptReco-v1/RECO and run=191277 -n 100 --conditions auto:com10 --python_filename=skim_HT.py
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
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring('/store/data/Run2012A/HT/RAW/v1/000/191/277/303840F2-3B87-E111-A2D2-003048F118D2.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/FE4AA5EA-1687-E111-B261-0025901D62A6.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/3C82778D-2C87-E111-9441-003048F118C2.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/EEC56DDF-2F87-E111-BE7F-BCAEC518FF54.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/B0DF3E61-3D87-E111-BEDC-003048D3750A.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/0EF0A120-3487-E111-BB3F-0025901D623C.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/C08B2CA6-2687-E111-B583-003048D3C980.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/8A782D2F-4087-E111-92D3-001D09F26509.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/D8CC4FDA-2087-E111-8518-003048D2C0F2.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/B8A732A0-1C87-E111-AA38-001D09F29619.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/DC863327-1987-E111-A2F8-001D09F24353.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/C2D10B30-2A87-E111-A796-001D09F25460.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/02DA2516-1E87-E111-AA7F-00237DDBE41A.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/2AE969C7-2D87-E111-BED5-0019B9F4A1D7.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/0C728E7D-2487-E111-B62F-003048D2BEA8.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/20424A4E-3187-E111-9C74-BCAEC518FF68.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/582F2798-2187-E111-949B-0025901D624A.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/F6B6DD53-1687-E111-87C2-BCAEC518FF6B.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/F69BE892-4687-E111-93C4-003048F11114.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/C4CE2AD7-2887-E111-BA5F-5404A63886C0.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/B22C5D36-2387-E111-BE3E-5404A63886EC.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/FE26B8F3-3787-E111-9943-BCAEC518FF50.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/6C2CF157-2787-E111-9522-001D09F34488.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/AAA73216-1E87-E111-AA21-003048F118D2.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/BEF737D5-3687-E111-857B-001D09F241B9.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/B6967A9B-4187-E111-9FC4-003048F118D2.root', 
        '/store/data/Run2012A/HT/RAW/v1/000/191/277/A832F6AE-1A87-E111-A32D-002481E0D7C0.root'),
    fileNames = cms.untracked.vstring('/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/F25D2E65-EC88-E111-8A9C-0025901D624A.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/F0B2F57E-0389-E111-8841-003048D37538.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/ECF0BC55-F488-E111-B6D8-0025901D5D90.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/DA441F75-0889-E111-9E5E-001D09F24024.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/D0343D09-0789-E111-A416-003048D3C90E.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/C4853D19-0289-E111-84DF-00237DDBE49C.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/C0D2F042-FF88-E111-9586-0015C5FDE067.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/B4EF60DA-EB88-E111-BF40-5404A6388694.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/B069075B-F488-E111-8A13-0025B32445E0.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/ACCF1F60-F288-E111-8333-BCAEC532971F.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/9C374337-0489-E111-BB65-001D09F2906A.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/9A3198E9-0489-E111-96CF-003048D3C982.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/94446BF0-0489-E111-B660-003048D373F6.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/92326171-FC88-E111-BDB6-E0CB4E4408E3.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/8E6AA219-0289-E111-9D0A-003048D37580.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/64FB661B-0289-E111-830E-0030486730C6.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/62939660-0189-E111-83E6-003048678110.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/5C496D99-0589-E111-8306-003048D37694.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/58B98D40-0489-E111-A24B-001D09F24399.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/38613142-FF88-E111-8A2D-001D09F2A49C.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/2C2DE1F9-FF88-E111-B127-003048D37456.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/260EA86F-FC88-E111-A9E6-003048D2BC52.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/22722F19-0289-E111-996D-001D09F28EA3.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/16C20ECD-0289-E111-A8FB-0030486780AC.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/1682161A-FD88-E111-B343-0025901D6268.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/12EB08AE-0C89-E111-8231-001D09F291D7.root', 
        '/store/data/Run2012A/HT/RECO/PromptReco-v1/000/191/277/02EC427A-F788-E111-B4D9-003048D2BB58.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.372.2.1 $'),
    annotation = cms.untracked.string('skims nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

# Additional output definition
process.SKIMStreamEXOHSCP = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('EXOHSCPPath', 
            'EXOHSCPDEDXPath')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep EventAux_*_*_*', 
        'keep LumiSummary_*_*_*', 
        'keep edmMergeableCounter_*_*_*', 
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
        'keep *_ak5PFJets_*_*', 
        'keep recoPFMETs_pfMet__*', 
        'keep recoBeamSpot_offlineBeamSpot__*'),
    fileName = cms.untracked.string('EXOHSCP.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EXOHSCP'),
        dataTier = cms.untracked.string('USER')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880)
)
process.SKIMStreamHighMET = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pfPath', 
            'tcPath')
    ),
    outputCommands = (cms.untracked.vstring('drop *', 'drop *', 'keep  FEDRawDataCollection_rawDataCollector_*_*', 'keep  FEDRawDataCollection_source_*_*', 'keep  FEDRawDataCollection_rawDataCollector_*_*', 'keep  FEDRawDataCollection_source_*_*', 'drop *_hlt*_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep FEDRawDataCollection_rawDataCollector_*_*', 'keep FEDRawDataCollection_source_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep DetIdedmEDCollection_siStripDigis_*_*', 'keep DetIdedmEDCollection_siPixelDigis_*_*', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_dt1DCosmicRecHits_*_*', 'keep *_dt4DCosmicSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep HBHERecHitsSorted_hbherecoMB_*_*', 'keep HORecHitsSorted_horecoMB_*_*', 'keep HFRecHitsSorted_hfrecoMB_*_*', 'keep ZDCDataFramesSorted_*Digis_*_*', 'keep ZDCRecHitsSorted_*_*_*', 'keep *_reducedHcalRecHits_*_*', 'keep *_castorreco_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalCompactTrigPrim_*_*', 'keep *_ecalTPSkim_*_*', 'keep *_selectDigi_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsEE_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsEB_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsES_*_*', 'keep *_hybridSuperClusters_*_*', 'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 'keep *_multi5x5SuperClusters_*_*', 'keep recoSuperClusters_multi5x5SuperClusters_*_*', 'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*', 'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*', 'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*', 'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*', 'drop recoClusterShapes_*_*_*', 'drop recoBasicClustersToOnerecoClusterShapesAssociation_*_*_*', 'drop recoBasicClusters_multi5x5BasicClusters_multi5x5BarrelBasicClusters_*', 'drop recoSuperClusters_multi5x5SuperClusters_multi5x5BarrelSuperClusters_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_electronMergedSeeds_*_*', 'keep recoGsfTracks_electronGsfTracks_*_*', 'keep recoGsfTrackExtras_electronGsfTracks_*_*', 'keep recoTrackExtras_electronGsfTracks_*_*', 'keep TrackingRecHitsOwned_electronGsfTracks_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoTrackExtras_generalTracks_*_*', 'keep TrackingRecHitsOwned_generalTracks_*_*', 'keep recoTracks_beamhaloTracks_*_*', 'keep recoTrackExtras_beamhaloTracks_*_*', 'keep TrackingRecHitsOwned_beamhaloTracks_*_*', 'keep recoTracks_regionalCosmicTracks_*_*', 'keep recoTrackExtras_regionalCosmicTracks_*_*', 'keep TrackingRecHitsOwned_regionalCosmicTracks_*_*', 'keep recoTracks_rsWithMaterialTracks_*_*', 'keep recoTrackExtras_rsWithMaterialTracks_*_*', 'keep TrackingRecHitsOwned_rsWithMaterialTracks_*_*', 'keep recoTracks_conversionStepTracks_*_*', 'keep recoTrackExtras_conversionStepTracks_*_*', 'keep TrackingRecHitsOwned_conversionStepTracks_*_*', 'keep *_ctfPixelLess_*_*', 'keep *_dedxTruncated40_*_*', 'keep *_dedxDiscrimASmi_*_*', 'keep *_dedxHarmonic2_*_*', 'keep *_trackExtrapolator_*_*', 'keep *_kt4CaloJets_*_*', 'keep *_kt6CaloJets_*_*', 'keep *_ak5CaloJets_*_*', 'keep *_ak7CaloJets_*_*', 'keep *_iterativeCone5CaloJets_*_*', 'keep *_iterativeCone15CaloJets_*_*', 'keep *_kt4PFJets_*_*', 'keep *_kt6PFJets_*_*', 'keep *_ak5PFJets_*_*', 'keep *_ak7PFJets_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_JetPlusTrackZSPCorJetAntiKt5_*_*', 'keep *_ak5TrackJets_*_*', 'keep *_kt4TrackJets_*_*', 'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_CastorTowerReco_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep *_ak5JetTracksAssociatorAtVertex_*_*', 'keep *_ak5JetTracksAssociatorAtCaloFace_*_*', 'keep *_ak5JetExtender_*_*', 'keep *_ak7JetTracksAssociatorAtVertex_*_*', 'keep *_ak7JetTracksAssociatorAtCaloFace_*_*', 'keep *_ak7JetExtender_*_*', 'keep *_ak5JetID_*_*', 'keep *_ak7JetID_*_*', 'keep *_ic5JetID_*_*', 'keep *_kt4JetID_*_*', 'keep *_kt6JetID_*_*', 'keep *_ak7BasicJets_*_*', 'keep *_ak7CastorJetID_*_*', 'keep double_kt6CaloJetsCentral_*_*', 'keep double_kt6PFJetsCentralChargedPileUp_*_*', 'keep double_kt6PFJetsCentralNeutral_*_*', 'keep double_kt6PFJetsCentralNeutralTight_*_*', 'keep *_fixedGridRho*_*_*', 'keep recoCaloMETs_met_*_*', 'keep recoCaloMETs_metNoHF_*_*', 'keep recoCaloMETs_metHO_*_*', 'keep recoCaloMETs_corMetGlobalMuons_*_*', 'keep recoCaloMETs_metNoHFHO_*_*', 'keep recoCaloMETs_metOptHO_*_*', 'keep recoCaloMETs_metOpt_*_*', 'keep recoCaloMETs_metOptNoHFHO_*_*', 'keep recoCaloMETs_metOptNoHF_*_*', 'keep recoMETs_htMetAK5_*_*', 'keep recoMETs_htMetAK7_*_*', 'keep recoMETs_htMetIC5_*_*', 'keep recoMETs_htMetKT4_*_*', 'keep recoMETs_htMetKT6_*_*', 'keep recoMETs_tcMet_*_*', 'keep recoMETs_tcMetWithPFclusters_*_*', 'keep recoPFMETs_pfMet_*_*', 'keep recoMuonMETCorrectionDataedmValueMap_muonMETValueMapProducer_*_*', 'keep recoMuonMETCorrectionDataedmValueMap_muonTCMETValueMapProducer_*_*', 'keep recoHcalNoiseRBXs_hcalnoise_*_*', 'keep HcalNoiseSummary_hcalnoise_*_*', 'keep *HaloData_*_*_*', 'keep *BeamHaloSummary_BeamHaloSummary_*_*', 'keep *_MuonSeed_*_*', 'keep *_ancientMuonSeed_*_*', 'keep *_mergedStandAloneMuonSeeds_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep TrackingRecHitsOwned_tevMuons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_CosmicMuonSeed_*_*', 'keep recoTrackExtras_cosmicMuons_*_*', 'keep TrackingRecHitsOwned_cosmicMuons_*_*', 'keep recoTrackExtras_globalCosmicMuons_*_*', 'keep TrackingRecHitsOwned_globalCosmicMuons_*_*', 'keep recoTrackExtras_cosmicMuons1Leg_*_*', 'keep TrackingRecHitsOwned_cosmicMuons1Leg_*_*', 'keep recoTrackExtras_globalCosmicMuons1Leg_*_*', 'keep TrackingRecHitsOwned_globalCosmicMuons1Leg_*_*', 'keep recoTracks_cosmicsVetoTracks_*_*', 'keep *_SETMuonSeed_*_*', 'keep recoTracks_standAloneSETMuons_*_*', 'keep recoTrackExtras_standAloneSETMuons_*_*', 'keep TrackingRecHitsOwned_standAloneSETMuons_*_*', 'keep recoTracks_globalSETMuons_*_*', 'keep recoTrackExtras_globalSETMuons_*_*', 'keep TrackingRecHitsOwned_globalSETMuons_*_*', 'keep recoMuons_muonsWithSET_*_*', 'keep *_muons_*_*', 'keep *_*_muons_*', 'drop *_muons_muons1stStep2muonsMap_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep recoTracks_tevMuons_*_*', 'keep recoTrackExtras_tevMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoTracksToOnerecoTracksAssociation_tevMuons_*_*', 'keep recoTracks_cosmicMuons_*_*', 'keep recoTracks_globalCosmicMuons_*_*', 'keep recoMuons_muonsFromCosmics_*_*', 'keep recoTracks_cosmicMuons1Leg_*_*', 'keep recoTracks_globalCosmicMuons1Leg_*_*', 'keep recoMuons_muonsFromCosmics1Leg_*_*', 'keep recoTracks_refittedStandAloneMuons_*_*', 'keep recoTrackExtras_refittedStandAloneMuons_*_*', 'keep TrackingRecHitsOwned_refittedStandAloneMuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_ghostTrackVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_simpleSecondaryVertexHighEffBJetTags_*_*', 'keep *_simpleSecondaryVertexHighPurBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_ghostTrackBJetTags_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronCands_*_*', 'keep *_softPFElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softElectronByIP3dBJetTags_*_*', 'keep *_softElectronByPtBJetTags_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonByIP3dBJetTags_*_*', 'keep *_softMuonByPtBJetTags_*_*', 'keep *_combinedMVABJetTags_*_*', 'keep *_ak5PFJetsRecoTauPiZeros_*_*', 'keep *_hpsPFTauProducer_*_*', 'keep *_hpsPFTauDiscrimination*_*_*', 'keep *_shrinkingConePFTauProducer_*_*', 'keep *_shrinkingConePFTauDiscrimination*_*_*', 'keep *_hpsTancTaus_*_*', 'keep *_hpsTancTausDiscrimination*_*_*', 'keep *_TCTauJetPlusTrackZSPCorJetAntiKt5_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep recoCaloTaus_caloRecoTauProducer*_*_*', 'keep *_caloRecoTauDiscrimination*_*_*', 'keep  *_offlinePrimaryVertices__*', 'keep  *_offlinePrimaryVerticesWithBS_*_*', 'keep  *_offlinePrimaryVerticesFromCosmicTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep *_generalV0Candidates_*_*', 'keep recoGsfElectronCores_gsfElectronCores_*_*', 'keep recoGsfElectrons_gsfElectrons_*_*', 'keep recoGsfElectronCores_uncleanedOnlyGsfElectronCores_*_*', 'keep recoGsfElectrons_uncleanedOnlyGsfElectrons_*_*', 'keep floatedmValueMap_eidRobustLoose_*_*', 'keep floatedmValueMap_eidRobustTight_*_*', 'keep floatedmValueMap_eidRobustHighEnergy_*_*', 'keep floatedmValueMap_eidLoose_*_*', 'keep floatedmValueMap_eidTight_*_*', 'keep recoPhotons_photons_*_*', 'keep recoPhotonCores_photonCore_*_*', 'keep recoConversions_conversions_*_*', 'drop *_conversions_uncleanedConversions_*', 'keep recoConversions_allConversions_*_*', 'keep recoTracks_ckfOutInTracksFromConversions_*_*')+cms.untracked.vstring('keep recoTracks_ckfInOutTracksFromConversions_*_*', 'keep recoTrackExtras_ckfOutInTracksFromConversions_*_*', 'keep recoTrackExtras_ckfInOutTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfOutInTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfInOutTracksFromConversions_*_*', 'keep recoConversions_uncleanedOnlyAllConversions_*_*', 'keep recoTracks_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 'keep recoTracks_uncleanedOnlyCkfInOutTracksFromConversions_*_*', 'keep recoTrackExtras_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 'keep recoTrackExtras_uncleanedOnlyCkfInOutTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_uncleanedOnlyCkfInOutTracksFromConversions_*_*', 'keep *_PhotonIDProd_*_*', 'keep *_hfRecoEcalCandidate_*_*', 'keep *_hfEMClusters_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHO_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHFEM_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHFHAD_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitHCAL_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitHO_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*', 'keep recoPFClusters_particleFlowClusterECAL_*_*', 'keep recoPFClusters_particleFlowClusterHCAL_*_*', 'keep recoPFClusters_particleFlowClusterHO_*_*', 'keep recoPFClusters_particleFlowClusterPS_*_*', 'keep recoPFBlocks_particleFlowBlock_*_*', 'keep recoPFCandidates_particleFlow_*_*', 'keep recoPFCandidates_particleFlowTmp_electrons_*', 'keep recoPFDisplacedVertexs_particleFlowDisplacedVertex_*_*', 'keep *_pfElectronTranslator_*_*', 'keep *_pfPhotonTranslator_*_*', 'keep *_particleFlow_electrons_*', 'keep *_particleFlow_photons_*', 'keep *_trackerDrivenElectronSeeds_preid_*', 'keep *_offlineBeamSpot_*_*', 'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtTriggerMenuLite_*_*', 'keep *_conditionsInEdm_*_*', 'keep *_l1extraParticles_*_*', 'keep *_l1L1GtObjectMap_*_*', 'keep L1MuGMTReadoutCollection_gtDigis_*_*', 'keep L1GctEmCand*_gctDigis_*_*', 'keep L1GctJetCand*_gctDigis_*_*', 'keep L1GctEtHad*_gctDigis_*_*', 'keep L1GctEtMiss*_gctDigis_*_*', 'keep L1GctEtTotal*_gctDigis_*_*', 'keep L1GctHtMiss*_gctDigis_*_*', 'keep L1GctJetCounts*_gctDigis_*_*', 'keep L1GctHFRingEtSums*_gctDigis_*_*', 'keep L1GctHFBitCounts*_gctDigis_*_*', 'keep LumiDetails_lumiProducer_*_*', 'keep LumiSummary_lumiProducer_*_*', 'drop *_hlt*_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep L1AcceptBunchCrossings_*_*_*', 'keep L1TriggerScalerss_*_*_*', 'keep Level1TriggerScalerss_*_*_*', 'keep LumiScalerss_*_*_*', 'keep BeamSpotOnlines_*_*_*', 'keep DcsStatuss_*_*_*', 'keep *_logErrorHarvester_*_*', 'drop *_MEtoEDMConverter_*_*', 'drop *_*_*_SKIM')),
    fileName = cms.untracked.string('HighMET.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('HighMET'),
        dataTier = cms.untracked.string('RAW-RECO')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880)
)

# Other statements
process.GlobalTag.globaltag = 'GR_R_52_V7::All'

# Path and EndPath definitions
process.SKIMStreamEXOHSCPOutPath = cms.EndPath(process.SKIMStreamEXOHSCP)
process.SKIMStreamHighMETOutPath = cms.EndPath(process.SKIMStreamHighMET)

# Schedule definition
process.schedule = cms.Schedule(process.EXOHSCPPath,process.EXOHSCPDEDXPath,process.pfPath,process.tcPath,process.SKIMStreamEXOHSCPOutPath,process.SKIMStreamHighMETOutPath)

