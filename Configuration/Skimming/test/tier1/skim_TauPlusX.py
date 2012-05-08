# Auto generated configuration file
# using: 
# Revision: 1.372.2.1 
# Source: /local/reps/CMSSW.admin/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: skims -s SKIM:@TauPlusX --data --no_exec --dbs find file,file.parent where dataset=/TauPlusX/Run2012A-PromptReco-v1/RECO and run=191277 -n 100 --conditions auto:com10 --python_filename=skim_TauPlusX.py
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
    secondaryFileNames = cms.untracked.vstring('/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/CE41A111-1E87-E111-8A2A-003048F117B6.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/303046E5-4087-E111-B2CA-001D09F23A20.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/2A7CDECE-2387-E111-A8F5-0025901D6288.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/CC59EDBB-3E87-E111-AADC-003048F117EC.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/E440E93C-1887-E111-947F-001D09F24D67.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/D8AD0FEC-3B87-E111-9BD1-001D09F252DA.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/88AF57E0-2F87-E111-BBCC-E0CB4E553651.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/E219A1C8-2087-E111-9D2C-001D09F2841C.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/80C4187F-2287-E111-B13E-0030486780B4.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/7AB35607-3287-E111-9BD9-003048F1C832.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/BA1EC5FF-4287-E111-8D4D-0025B32035BC.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/221B7911-1E87-E111-BDBB-003048F024FE.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/B06D35EF-1B87-E111-97A2-0015C5FDE067.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/56456FD1-3687-E111-8DE2-5404A638869B.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/3C0EFD53-2787-E111-BEC1-001D09F2A690.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/CAABDE8B-4687-E111-A59B-001D09F2906A.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/226E6A20-3487-E111-BE0E-00215AEDFD74.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/F0429DE5-2587-E111-82BD-001D09F24D67.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/6227AA2D-2A87-E111-B09F-0019B9F4A1D7.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/866B4DBF-2D87-E111-87B5-5404A638868F.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/3063539C-2B87-E111-A9BD-001D09F25460.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/B48D81AE-1987-E111-9354-0030486780EC.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/329EE63B-1687-E111-A31C-003048D2C1C4.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/3A9A61FD-3987-E111-A004-003048F01E88.root', 
        '/store/data/Run2012A/TauPlusX/RAW/v1/000/191/277/3E34B0C2-2887-E111-B23C-003048D37456.root'),
    fileNames = cms.untracked.vstring('/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/F2D42EE4-EB88-E111-BD55-5404A63886CC.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/E04DD4E5-F688-E111-AB5E-001D09F241B9.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/D0E7BB6C-EA88-E111-9A1D-0025B32036D2.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/D083EFF3-E888-E111-B90F-002481E0DEC6.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/AC3CFC73-0889-E111-BA3B-001D09F34488.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/A43B88F2-E888-E111-AF30-0025B320384C.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/9AD3B78F-E388-E111-BD18-003048D3C944.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/9875BFE3-F688-E111-9556-003048D37560.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/9828D859-F688-E111-B876-BCAEC532971D.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/94680E48-F588-E111-809C-BCAEC5364CED.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/8EF33005-F088-E111-AF17-5404A638869B.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/88E7FCCE-FD88-E111-9E9D-003048D37560.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/864ADEB5-ED88-E111-9D01-0025901D5C88.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/8265C683-E988-E111-B1DF-BCAEC532971D.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/7CFA7C74-0889-E111-9E94-001D09F2906A.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/7A8EA4E5-F688-E111-B2AF-001D09F2910A.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/7006C26B-EA88-E111-9FBC-002481E0D524.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/56F8CB55-EC88-E111-95EE-0025901D6272.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/4AFDB3AB-0089-E111-BCFB-001D09F242EF.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/4AA0B374-0889-E111-97BC-001D09F252DA.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/4A2FC1F4-F588-E111-B0C5-001D09F2932B.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/445D9CCB-0289-E111-B440-003048D37694.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/2CE4EBAA-FB88-E111-AE23-001D09F2424A.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/1E3975E4-F688-E111-A931-001D09F27067.root', 
        '/store/data/Run2012A/TauPlusX/RECO/PromptReco-v1/000/191/277/1A991DE5-F688-E111-8A3E-003048D2C0F2.root')
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
process.SKIMStreamMuTauMET = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('mutauMETSkimPath')
    ),
    outputCommands = (cms.untracked.vstring('drop *', 'drop *', 'keep  FEDRawDataCollection_rawDataCollector_*_*', 'keep  FEDRawDataCollection_source_*_*', 'keep  FEDRawDataCollection_rawDataCollector_*_*', 'keep  FEDRawDataCollection_source_*_*', 'drop *_hlt*_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep FEDRawDataCollection_rawDataCollector_*_*', 'keep FEDRawDataCollection_source_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep DetIdedmEDCollection_siStripDigis_*_*', 'keep DetIdedmEDCollection_siPixelDigis_*_*', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_dt1DCosmicRecHits_*_*', 'keep *_dt4DCosmicSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep HBHERecHitsSorted_hbherecoMB_*_*', 'keep HORecHitsSorted_horecoMB_*_*', 'keep HFRecHitsSorted_hfrecoMB_*_*', 'keep ZDCDataFramesSorted_*Digis_*_*', 'keep ZDCRecHitsSorted_*_*_*', 'keep *_reducedHcalRecHits_*_*', 'keep *_castorreco_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalCompactTrigPrim_*_*', 'keep *_ecalTPSkim_*_*', 'keep *_selectDigi_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsEE_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsEB_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsES_*_*', 'keep *_hybridSuperClusters_*_*', 'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 'keep *_multi5x5SuperClusters_*_*', 'keep recoSuperClusters_multi5x5SuperClusters_*_*', 'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*', 'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*', 'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*', 'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*', 'drop recoClusterShapes_*_*_*', 'drop recoBasicClustersToOnerecoClusterShapesAssociation_*_*_*', 'drop recoBasicClusters_multi5x5BasicClusters_multi5x5BarrelBasicClusters_*', 'drop recoSuperClusters_multi5x5SuperClusters_multi5x5BarrelSuperClusters_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_electronMergedSeeds_*_*', 'keep recoGsfTracks_electronGsfTracks_*_*', 'keep recoGsfTrackExtras_electronGsfTracks_*_*', 'keep recoTrackExtras_electronGsfTracks_*_*', 'keep TrackingRecHitsOwned_electronGsfTracks_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoTrackExtras_generalTracks_*_*', 'keep TrackingRecHitsOwned_generalTracks_*_*', 'keep recoTracks_beamhaloTracks_*_*', 'keep recoTrackExtras_beamhaloTracks_*_*', 'keep TrackingRecHitsOwned_beamhaloTracks_*_*', 'keep recoTracks_regionalCosmicTracks_*_*', 'keep recoTrackExtras_regionalCosmicTracks_*_*', 'keep TrackingRecHitsOwned_regionalCosmicTracks_*_*', 'keep recoTracks_rsWithMaterialTracks_*_*', 'keep recoTrackExtras_rsWithMaterialTracks_*_*', 'keep TrackingRecHitsOwned_rsWithMaterialTracks_*_*', 'keep recoTracks_conversionStepTracks_*_*', 'keep recoTrackExtras_conversionStepTracks_*_*', 'keep TrackingRecHitsOwned_conversionStepTracks_*_*', 'keep *_ctfPixelLess_*_*', 'keep *_dedxTruncated40_*_*', 'keep *_dedxDiscrimASmi_*_*', 'keep *_dedxHarmonic2_*_*', 'keep *_trackExtrapolator_*_*', 'keep *_kt4CaloJets_*_*', 'keep *_kt6CaloJets_*_*', 'keep *_ak5CaloJets_*_*', 'keep *_ak7CaloJets_*_*', 'keep *_iterativeCone5CaloJets_*_*', 'keep *_iterativeCone15CaloJets_*_*', 'keep *_kt4PFJets_*_*', 'keep *_kt6PFJets_*_*', 'keep *_ak5PFJets_*_*', 'keep *_ak7PFJets_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_JetPlusTrackZSPCorJetAntiKt5_*_*', 'keep *_ak5TrackJets_*_*', 'keep *_kt4TrackJets_*_*', 'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_CastorTowerReco_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep *_ak5JetTracksAssociatorAtVertex_*_*', 'keep *_ak5JetTracksAssociatorAtCaloFace_*_*', 'keep *_ak5JetExtender_*_*', 'keep *_ak7JetTracksAssociatorAtVertex_*_*', 'keep *_ak7JetTracksAssociatorAtCaloFace_*_*', 'keep *_ak7JetExtender_*_*', 'keep *_ak5JetID_*_*', 'keep *_ak7JetID_*_*', 'keep *_ic5JetID_*_*', 'keep *_kt4JetID_*_*', 'keep *_kt6JetID_*_*', 'keep *_ak7BasicJets_*_*', 'keep *_ak7CastorJetID_*_*', 'keep double_kt6CaloJetsCentral_*_*', 'keep double_kt6PFJetsCentralChargedPileUp_*_*', 'keep double_kt6PFJetsCentralNeutral_*_*', 'keep double_kt6PFJetsCentralNeutralTight_*_*', 'keep *_fixedGridRho*_*_*', 'keep recoCaloMETs_met_*_*', 'keep recoCaloMETs_metNoHF_*_*', 'keep recoCaloMETs_metHO_*_*', 'keep recoCaloMETs_corMetGlobalMuons_*_*', 'keep recoCaloMETs_metNoHFHO_*_*', 'keep recoCaloMETs_metOptHO_*_*', 'keep recoCaloMETs_metOpt_*_*', 'keep recoCaloMETs_metOptNoHFHO_*_*', 'keep recoCaloMETs_metOptNoHF_*_*', 'keep recoMETs_htMetAK5_*_*', 'keep recoMETs_htMetAK7_*_*', 'keep recoMETs_htMetIC5_*_*', 'keep recoMETs_htMetKT4_*_*', 'keep recoMETs_htMetKT6_*_*', 'keep recoMETs_tcMet_*_*', 'keep recoMETs_tcMetWithPFclusters_*_*', 'keep recoPFMETs_pfMet_*_*', 'keep recoMuonMETCorrectionDataedmValueMap_muonMETValueMapProducer_*_*', 'keep recoMuonMETCorrectionDataedmValueMap_muonTCMETValueMapProducer_*_*', 'keep recoHcalNoiseRBXs_hcalnoise_*_*', 'keep HcalNoiseSummary_hcalnoise_*_*', 'keep *HaloData_*_*_*', 'keep *BeamHaloSummary_BeamHaloSummary_*_*', 'keep *_MuonSeed_*_*', 'keep *_ancientMuonSeed_*_*', 'keep *_mergedStandAloneMuonSeeds_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep TrackingRecHitsOwned_tevMuons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_CosmicMuonSeed_*_*', 'keep recoTrackExtras_cosmicMuons_*_*', 'keep TrackingRecHitsOwned_cosmicMuons_*_*', 'keep recoTrackExtras_globalCosmicMuons_*_*', 'keep TrackingRecHitsOwned_globalCosmicMuons_*_*', 'keep recoTrackExtras_cosmicMuons1Leg_*_*', 'keep TrackingRecHitsOwned_cosmicMuons1Leg_*_*', 'keep recoTrackExtras_globalCosmicMuons1Leg_*_*', 'keep TrackingRecHitsOwned_globalCosmicMuons1Leg_*_*', 'keep recoTracks_cosmicsVetoTracks_*_*', 'keep *_SETMuonSeed_*_*', 'keep recoTracks_standAloneSETMuons_*_*', 'keep recoTrackExtras_standAloneSETMuons_*_*', 'keep TrackingRecHitsOwned_standAloneSETMuons_*_*', 'keep recoTracks_globalSETMuons_*_*', 'keep recoTrackExtras_globalSETMuons_*_*', 'keep TrackingRecHitsOwned_globalSETMuons_*_*', 'keep recoMuons_muonsWithSET_*_*', 'keep *_muons_*_*', 'keep *_*_muons_*', 'drop *_muons_muons1stStep2muonsMap_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep recoTracks_tevMuons_*_*', 'keep recoTrackExtras_tevMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoTracksToOnerecoTracksAssociation_tevMuons_*_*', 'keep recoTracks_cosmicMuons_*_*', 'keep recoTracks_globalCosmicMuons_*_*', 'keep recoMuons_muonsFromCosmics_*_*', 'keep recoTracks_cosmicMuons1Leg_*_*', 'keep recoTracks_globalCosmicMuons1Leg_*_*', 'keep recoMuons_muonsFromCosmics1Leg_*_*', 'keep recoTracks_refittedStandAloneMuons_*_*', 'keep recoTrackExtras_refittedStandAloneMuons_*_*', 'keep TrackingRecHitsOwned_refittedStandAloneMuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_ghostTrackVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_simpleSecondaryVertexHighEffBJetTags_*_*', 'keep *_simpleSecondaryVertexHighPurBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_ghostTrackBJetTags_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronCands_*_*', 'keep *_softPFElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softElectronByIP3dBJetTags_*_*', 'keep *_softElectronByPtBJetTags_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonByIP3dBJetTags_*_*', 'keep *_softMuonByPtBJetTags_*_*', 'keep *_combinedMVABJetTags_*_*', 'keep *_ak5PFJetsRecoTauPiZeros_*_*', 'keep *_hpsPFTauProducer_*_*', 'keep *_hpsPFTauDiscrimination*_*_*', 'keep *_shrinkingConePFTauProducer_*_*', 'keep *_shrinkingConePFTauDiscrimination*_*_*', 'keep *_hpsTancTaus_*_*', 'keep *_hpsTancTausDiscrimination*_*_*', 'keep *_TCTauJetPlusTrackZSPCorJetAntiKt5_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep recoCaloTaus_caloRecoTauProducer*_*_*', 'keep *_caloRecoTauDiscrimination*_*_*', 'keep  *_offlinePrimaryVertices__*', 'keep  *_offlinePrimaryVerticesWithBS_*_*', 'keep  *_offlinePrimaryVerticesFromCosmicTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep *_generalV0Candidates_*_*', 'keep recoGsfElectronCores_gsfElectronCores_*_*', 'keep recoGsfElectrons_gsfElectrons_*_*', 'keep recoGsfElectronCores_uncleanedOnlyGsfElectronCores_*_*', 'keep recoGsfElectrons_uncleanedOnlyGsfElectrons_*_*', 'keep floatedmValueMap_eidRobustLoose_*_*', 'keep floatedmValueMap_eidRobustTight_*_*', 'keep floatedmValueMap_eidRobustHighEnergy_*_*', 'keep floatedmValueMap_eidLoose_*_*', 'keep floatedmValueMap_eidTight_*_*', 'keep recoPhotons_photons_*_*', 'keep recoPhotonCores_photonCore_*_*', 'keep recoConversions_conversions_*_*', 'drop *_conversions_uncleanedConversions_*', 'keep recoConversions_allConversions_*_*', 'keep recoTracks_ckfOutInTracksFromConversions_*_*')+cms.untracked.vstring('keep recoTracks_ckfInOutTracksFromConversions_*_*', 'keep recoTrackExtras_ckfOutInTracksFromConversions_*_*', 'keep recoTrackExtras_ckfInOutTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfOutInTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfInOutTracksFromConversions_*_*', 'keep recoConversions_uncleanedOnlyAllConversions_*_*', 'keep recoTracks_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 'keep recoTracks_uncleanedOnlyCkfInOutTracksFromConversions_*_*', 'keep recoTrackExtras_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 'keep recoTrackExtras_uncleanedOnlyCkfInOutTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_uncleanedOnlyCkfInOutTracksFromConversions_*_*', 'keep *_PhotonIDProd_*_*', 'keep *_hfRecoEcalCandidate_*_*', 'keep *_hfEMClusters_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHO_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHFEM_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHFHAD_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitHCAL_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitHO_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*', 'keep recoPFClusters_particleFlowClusterECAL_*_*', 'keep recoPFClusters_particleFlowClusterHCAL_*_*', 'keep recoPFClusters_particleFlowClusterHO_*_*', 'keep recoPFClusters_particleFlowClusterPS_*_*', 'keep recoPFBlocks_particleFlowBlock_*_*', 'keep recoPFCandidates_particleFlow_*_*', 'keep recoPFCandidates_particleFlowTmp_electrons_*', 'keep recoPFDisplacedVertexs_particleFlowDisplacedVertex_*_*', 'keep *_pfElectronTranslator_*_*', 'keep *_pfPhotonTranslator_*_*', 'keep *_particleFlow_electrons_*', 'keep *_particleFlow_photons_*', 'keep *_trackerDrivenElectronSeeds_preid_*', 'keep *_offlineBeamSpot_*_*', 'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtTriggerMenuLite_*_*', 'keep *_conditionsInEdm_*_*', 'keep *_l1extraParticles_*_*', 'keep *_l1L1GtObjectMap_*_*', 'keep L1MuGMTReadoutCollection_gtDigis_*_*', 'keep L1GctEmCand*_gctDigis_*_*', 'keep L1GctJetCand*_gctDigis_*_*', 'keep L1GctEtHad*_gctDigis_*_*', 'keep L1GctEtMiss*_gctDigis_*_*', 'keep L1GctEtTotal*_gctDigis_*_*', 'keep L1GctHtMiss*_gctDigis_*_*', 'keep L1GctJetCounts*_gctDigis_*_*', 'keep L1GctHFRingEtSums*_gctDigis_*_*', 'keep L1GctHFBitCounts*_gctDigis_*_*', 'keep LumiDetails_lumiProducer_*_*', 'keep LumiSummary_lumiProducer_*_*', 'drop *_hlt*_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep L1AcceptBunchCrossings_*_*_*', 'keep L1TriggerScalerss_*_*_*', 'keep Level1TriggerScalerss_*_*_*', 'keep LumiScalerss_*_*_*', 'keep BeamSpotOnlines_*_*_*', 'keep DcsStatuss_*_*_*', 'keep *_logErrorHarvester_*_*', 'drop *_MEtoEDMConverter_*_*', 'drop *_*_*_SKIM')),
    fileName = cms.untracked.string('MuTauMET.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('MuTauMET'),
        dataTier = cms.untracked.string('RAW-RECO')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880)
)

# Other statements
process.GlobalTag.globaltag = 'GR_R_52_V7::All'

# Path and EndPath definitions
process.SKIMStreamMuTauMETOutPath = cms.EndPath(process.SKIMStreamMuTauMET)

# Schedule definition
process.schedule = cms.Schedule(process.mutauMETSkimPath,process.SKIMStreamMuTauMETOutPath)

