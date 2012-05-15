# Auto generated configuration file
# using: 
# Revision: 1.372.2.1 
# Source: /local/reps/CMSSW.admin/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: skims -s SKIM:@MuOnia --data --no_exec --dbs find file,file.parent where dataset=/MuOnia/Run2012A-PromptReco-v1/RECO and run=191277 -n 100 --conditions auto:com10 --python_filename=skim_MuOnia.py
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
    secondaryFileNames = cms.untracked.vstring('/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/981DFA59-1D87-E111-A973-003048F1C58C.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/0411FF24-3487-E111-93E6-5404A63886A8.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/0A6DFA91-4187-E111-8B7A-001D09F2AD4D.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/90160054-2787-E111-9719-001D09F25460.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/8C10EB5E-2C87-E111-A7F8-003048678110.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/BC2E3304-2D87-E111-A4E5-0025901D6272.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/44DFB405-3A87-E111-A8D0-001D09F29533.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/16BB12B6-1987-E111-BE8A-E0CB4E4408E3.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/7AB43161-3D87-E111-B87A-00237DDC5C24.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/4A5E2884-2287-E111-B490-003048D2BC38.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/4A3AFB78-2987-E111-8C58-0025901D629C.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/C0877D97-1A87-E111-9568-5404A63886C5.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/200BF0CE-2387-E111-B432-5404A63886C4.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/623C6DE1-2F87-E111-A47B-BCAEC53296FB.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/FC578A11-1E87-E111-B46C-003048F118C2.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/5861E44A-1687-E111-BA1E-003048D2BA82.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/EABBB827-1987-E111-AD00-001D09F29321.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/46E74211-1E87-E111-A53D-003048F1C58C.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/ECDF49E6-1687-E111-A76B-001D09F253D4.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/086416A9-3787-E111-A952-003048F024FA.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/A83B392E-4087-E111-BEFC-001D09F24D8A.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/DAF3505A-2B87-E111-8066-003048F1183E.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/52759870-2E87-E111-93A4-0025901D629C.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/86075740-2587-E111-A9EB-001D09F291D7.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/44DC77DC-2087-E111-9F05-001D09F24024.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/AE0F021C-2887-E111-B67C-0025901D626C.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/E4F2B497-2187-E111-86F0-0025901D624E.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/02D913A8-2687-E111-ADDB-003048D2BC42.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/D8344BF2-3B87-E111-85AA-0019B9F4A1D7.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/B0FDA0BC-3E87-E111-B191-003048D3C932.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/54ABFB03-3287-E111-9C1F-0025901D626C.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/DC1722D2-3487-E111-AD7B-5404A63886A8.root', 
        '/store/data/Run2012A/MuOnia/RAW/v1/000/191/277/A2F6AE98-4687-E111-9042-003048F024C2.root'),
    fileNames = cms.untracked.vstring('/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/FAB39442-0489-E111-BBA0-0019B9F72D71.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/F2143317-E888-E111-A8CD-BCAEC518FF50.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/EAF51CC2-ED88-E111-AADD-BCAEC532971C.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/DEFCB032-F888-E111-87A7-002481E0D646.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/D424486B-EC88-E111-9688-001D09F2A690.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/CE6A36D1-FD88-E111-A466-0025901D5DEE.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/CA98143B-FF88-E111-8A69-E0CB4E553673.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/CA2ED6AF-0089-E111-BCDE-BCAEC5329709.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/C61F2936-EF88-E111-B94F-BCAEC5364CED.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/C4202EAD-0089-E111-96DD-BCAEC5364C62.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/C040F022-FD88-E111-8EF6-001D09F295A1.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/BC9D2480-0389-E111-948A-003048D375AA.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/B65A893D-F388-E111-B15C-0030486780E6.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/B287966A-EA88-E111-BE5E-BCAEC518FF89.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/A270B726-FD88-E111-8C83-003048D2C0F2.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/A03C1FB0-0089-E111-956C-485B3962633D.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/96B49258-F488-E111-9FAF-0025901D5C80.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/9285C46B-EC88-E111-8A6D-001D09F29619.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/723257AE-0C89-E111-8AF6-001D09F276CF.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/6E6F036B-FC88-E111-8AC3-003048D37456.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/6665C206-E688-E111-9B74-003048D2C16E.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/4C4D23BC-ED88-E111-BB63-E0CB4E4408C4.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/4AD350AC-FB88-E111-B99D-003048678110.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/3641C840-0489-E111-9C5F-001D09F23A20.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/32B21A8C-FE88-E111-99FA-001D09F290CE.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/2EF4DD3B-F388-E111-AA46-001D09F29597.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/2EC41722-F488-E111-9DFC-5404A63886B6.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/2A912AF9-FF88-E111-A2B2-0025901D5DB2.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/26171569-EC88-E111-8A48-003048D3C982.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/1EA4E645-F588-E111-9689-0025901D624A.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/1027C0D5-F888-E111-BCAD-0030486780AC.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/0C3A907A-F788-E111-BB47-003048D2BEA8.root', 
        '/store/data/Run2012A/MuOnia/RECO/PromptReco-v1/000/191/277/02361625-FD88-E111-89F7-0025B320384C.root')
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
process.SKIMStreamChiB = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('upsilonHLTPath')
    ),
    outputCommands = (cms.untracked.vstring('drop *', 'keep DetIdedmEDCollection_siStripDigis_*_*', 'keep DetIdedmEDCollection_siPixelDigis_*_*', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_dt1DCosmicRecHits_*_*', 'keep *_dt4DCosmicSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep HBHERecHitsSorted_hbherecoMB_*_*', 'keep HORecHitsSorted_horecoMB_*_*', 'keep HFRecHitsSorted_hfrecoMB_*_*', 'keep ZDCDataFramesSorted_*Digis_*_*', 'keep ZDCRecHitsSorted_*_*_*', 'keep *_reducedHcalRecHits_*_*', 'keep *_castorreco_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalCompactTrigPrim_*_*', 'keep *_ecalTPSkim_*_*', 'keep *_selectDigi_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsEE_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsEB_*_*', 'keep EcalRecHitsSorted_reducedEcalRecHitsES_*_*', 'keep *_hybridSuperClusters_*_*', 'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 'keep *_multi5x5SuperClusters_*_*', 'keep recoSuperClusters_multi5x5SuperClusters_*_*', 'keep recoSuperClusters_multi5x5SuperClustersWithPreshower_*_*', 'keep recoSuperClusters_correctedMulti5x5SuperClustersWithPreshower_*_*', 'keep recoPreshowerClusters_multi5x5SuperClustersWithPreshower_*_*', 'keep recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_*_*', 'drop recoClusterShapes_*_*_*', 'drop recoBasicClustersToOnerecoClusterShapesAssociation_*_*_*', 'drop recoBasicClusters_multi5x5BasicClusters_multi5x5BarrelBasicClusters_*', 'drop recoSuperClusters_multi5x5SuperClusters_multi5x5BarrelSuperClusters_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_electronMergedSeeds_*_*', 'keep recoGsfTracks_electronGsfTracks_*_*', 'keep recoGsfTrackExtras_electronGsfTracks_*_*', 'keep recoTrackExtras_electronGsfTracks_*_*', 'keep TrackingRecHitsOwned_electronGsfTracks_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoTrackExtras_generalTracks_*_*', 'keep TrackingRecHitsOwned_generalTracks_*_*', 'keep recoTracks_beamhaloTracks_*_*', 'keep recoTrackExtras_beamhaloTracks_*_*', 'keep TrackingRecHitsOwned_beamhaloTracks_*_*', 'keep recoTracks_regionalCosmicTracks_*_*', 'keep recoTrackExtras_regionalCosmicTracks_*_*', 'keep TrackingRecHitsOwned_regionalCosmicTracks_*_*', 'keep recoTracks_rsWithMaterialTracks_*_*', 'keep recoTrackExtras_rsWithMaterialTracks_*_*', 'keep TrackingRecHitsOwned_rsWithMaterialTracks_*_*', 'keep recoTracks_conversionStepTracks_*_*', 'keep recoTrackExtras_conversionStepTracks_*_*', 'keep TrackingRecHitsOwned_conversionStepTracks_*_*', 'keep *_ctfPixelLess_*_*', 'keep *_dedxTruncated40_*_*', 'keep *_dedxDiscrimASmi_*_*', 'keep *_dedxHarmonic2_*_*', 'keep *_trackExtrapolator_*_*', 'keep *_kt4CaloJets_*_*', 'keep *_kt6CaloJets_*_*', 'keep *_ak5CaloJets_*_*', 'keep *_ak7CaloJets_*_*', 'keep *_iterativeCone5CaloJets_*_*', 'keep *_iterativeCone15CaloJets_*_*', 'keep *_kt4PFJets_*_*', 'keep *_kt6PFJets_*_*', 'keep *_ak5PFJets_*_*', 'keep *_ak7PFJets_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_JetPlusTrackZSPCorJetAntiKt5_*_*', 'keep *_ak5TrackJets_*_*', 'keep *_kt4TrackJets_*_*', 'keep recoRecoChargedRefCandidates_trackRefsForJets_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_CastorTowerReco_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep *_ak5JetTracksAssociatorAtVertex_*_*', 'keep *_ak5JetTracksAssociatorAtCaloFace_*_*', 'keep *_ak5JetExtender_*_*', 'keep *_ak7JetTracksAssociatorAtVertex_*_*', 'keep *_ak7JetTracksAssociatorAtCaloFace_*_*', 'keep *_ak7JetExtender_*_*', 'keep *_ak5JetID_*_*', 'keep *_ak7JetID_*_*', 'keep *_ic5JetID_*_*', 'keep *_kt4JetID_*_*', 'keep *_kt6JetID_*_*', 'keep *_ak7BasicJets_*_*', 'keep *_ak7CastorJetID_*_*', 'keep double_kt6CaloJetsCentral_*_*', 'keep double_kt6PFJetsCentralChargedPileUp_*_*', 'keep double_kt6PFJetsCentralNeutral_*_*', 'keep double_kt6PFJetsCentralNeutralTight_*_*', 'keep *_fixedGridRho*_*_*', 'keep recoCaloMETs_met_*_*', 'keep recoCaloMETs_metNoHF_*_*', 'keep recoCaloMETs_metHO_*_*', 'keep recoCaloMETs_corMetGlobalMuons_*_*', 'keep recoCaloMETs_metNoHFHO_*_*', 'keep recoCaloMETs_metOptHO_*_*', 'keep recoCaloMETs_metOpt_*_*', 'keep recoCaloMETs_metOptNoHFHO_*_*', 'keep recoCaloMETs_metOptNoHF_*_*', 'keep recoMETs_htMetAK5_*_*', 'keep recoMETs_htMetAK7_*_*', 'keep recoMETs_htMetIC5_*_*', 'keep recoMETs_htMetKT4_*_*', 'keep recoMETs_htMetKT6_*_*', 'keep recoMETs_tcMet_*_*', 'keep recoMETs_tcMetWithPFclusters_*_*', 'keep recoPFMETs_pfMet_*_*', 'keep recoMuonMETCorrectionDataedmValueMap_muonMETValueMapProducer_*_*', 'keep recoMuonMETCorrectionDataedmValueMap_muonTCMETValueMapProducer_*_*', 'keep recoHcalNoiseRBXs_hcalnoise_*_*', 'keep HcalNoiseSummary_hcalnoise_*_*', 'keep *HaloData_*_*_*', 'keep *BeamHaloSummary_BeamHaloSummary_*_*', 'keep *_MuonSeed_*_*', 'keep *_ancientMuonSeed_*_*', 'keep *_mergedStandAloneMuonSeeds_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep TrackingRecHitsOwned_tevMuons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_CosmicMuonSeed_*_*', 'keep recoTrackExtras_cosmicMuons_*_*', 'keep TrackingRecHitsOwned_cosmicMuons_*_*', 'keep recoTrackExtras_globalCosmicMuons_*_*', 'keep TrackingRecHitsOwned_globalCosmicMuons_*_*', 'keep recoTrackExtras_cosmicMuons1Leg_*_*', 'keep TrackingRecHitsOwned_cosmicMuons1Leg_*_*', 'keep recoTrackExtras_globalCosmicMuons1Leg_*_*', 'keep TrackingRecHitsOwned_globalCosmicMuons1Leg_*_*', 'keep recoTracks_cosmicsVetoTracks_*_*', 'keep *_SETMuonSeed_*_*', 'keep recoTracks_standAloneSETMuons_*_*', 'keep recoTrackExtras_standAloneSETMuons_*_*', 'keep TrackingRecHitsOwned_standAloneSETMuons_*_*', 'keep recoTracks_globalSETMuons_*_*', 'keep recoTrackExtras_globalSETMuons_*_*', 'keep TrackingRecHitsOwned_globalSETMuons_*_*', 'keep recoMuons_muonsWithSET_*_*', 'keep *_muons_*_*', 'keep *_*_muons_*', 'drop *_muons_muons1stStep2muonsMap_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep recoTracks_tevMuons_*_*', 'keep recoTrackExtras_tevMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoTracksToOnerecoTracksAssociation_tevMuons_*_*', 'keep recoTracks_cosmicMuons_*_*', 'keep recoTracks_globalCosmicMuons_*_*', 'keep recoMuons_muonsFromCosmics_*_*', 'keep recoTracks_cosmicMuons1Leg_*_*', 'keep recoTracks_globalCosmicMuons1Leg_*_*', 'keep recoMuons_muonsFromCosmics1Leg_*_*', 'keep recoTracks_refittedStandAloneMuons_*_*', 'keep recoTrackExtras_refittedStandAloneMuons_*_*', 'keep TrackingRecHitsOwned_refittedStandAloneMuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_ghostTrackVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_simpleSecondaryVertexHighEffBJetTags_*_*', 'keep *_simpleSecondaryVertexHighPurBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_ghostTrackBJetTags_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronCands_*_*', 'keep *_softPFElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softElectronByIP3dBJetTags_*_*', 'keep *_softElectronByPtBJetTags_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonByIP3dBJetTags_*_*', 'keep *_softMuonByPtBJetTags_*_*', 'keep *_combinedMVABJetTags_*_*', 'keep *_ak5PFJetsRecoTauPiZeros_*_*', 'keep *_hpsPFTauProducer_*_*', 'keep *_hpsPFTauDiscrimination*_*_*', 'keep *_shrinkingConePFTauProducer_*_*', 'keep *_shrinkingConePFTauDiscrimination*_*_*', 'keep *_hpsTancTaus_*_*', 'keep *_hpsTancTausDiscrimination*_*_*', 'keep *_TCTauJetPlusTrackZSPCorJetAntiKt5_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep recoCaloTaus_caloRecoTauProducer*_*_*', 'keep *_caloRecoTauDiscrimination*_*_*', 'keep  *_offlinePrimaryVertices__*', 'keep  *_offlinePrimaryVerticesWithBS_*_*', 'keep  *_offlinePrimaryVerticesFromCosmicTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep *_generalV0Candidates_*_*', 'keep recoGsfElectronCores_gsfElectronCores_*_*', 'keep recoGsfElectrons_gsfElectrons_*_*', 'keep recoGsfElectronCores_uncleanedOnlyGsfElectronCores_*_*', 'keep recoGsfElectrons_uncleanedOnlyGsfElectrons_*_*', 'keep floatedmValueMap_eidRobustLoose_*_*', 'keep floatedmValueMap_eidRobustTight_*_*', 'keep floatedmValueMap_eidRobustHighEnergy_*_*', 'keep floatedmValueMap_eidLoose_*_*', 'keep floatedmValueMap_eidTight_*_*', 'keep recoPhotons_photons_*_*', 'keep recoPhotonCores_photonCore_*_*', 'keep recoConversions_conversions_*_*', 'drop *_conversions_uncleanedConversions_*', 'keep recoConversions_allConversions_*_*', 'keep recoTracks_ckfOutInTracksFromConversions_*_*', 'keep recoTracks_ckfInOutTracksFromConversions_*_*', 'keep recoTrackExtras_ckfOutInTracksFromConversions_*_*', 'keep recoTrackExtras_ckfInOutTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfOutInTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfInOutTracksFromConversions_*_*', 'keep recoConversions_uncleanedOnlyAllConversions_*_*', 'keep recoTracks_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 'keep recoTracks_uncleanedOnlyCkfInOutTracksFromConversions_*_*', 'keep recoTrackExtras_uncleanedOnlyCkfOutInTracksFromConversions_*_*', 'keep recoTrackExtras_uncleanedOnlyCkfInOutTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_uncleanedOnlyCkfOutInTracksFromConversions_*_*')+cms.untracked.vstring('keep TrackingRecHitsOwned_uncleanedOnlyCkfInOutTracksFromConversions_*_*', 'keep *_PhotonIDProd_*_*', 'keep *_hfRecoEcalCandidate_*_*', 'keep *_hfEMClusters_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHO_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHFEM_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterHFHAD_Cleaned_*', 'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitHCAL_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitHO_Cleaned_*', 'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*', 'keep recoPFClusters_particleFlowClusterECAL_*_*', 'keep recoPFClusters_particleFlowClusterHCAL_*_*', 'keep recoPFClusters_particleFlowClusterHO_*_*', 'keep recoPFClusters_particleFlowClusterPS_*_*', 'keep recoPFBlocks_particleFlowBlock_*_*', 'keep recoPFCandidates_particleFlow_*_*', 'keep recoPFCandidates_particleFlowTmp_electrons_*', 'keep recoPFDisplacedVertexs_particleFlowDisplacedVertex_*_*', 'keep *_pfElectronTranslator_*_*', 'keep *_pfPhotonTranslator_*_*', 'keep *_particleFlow_electrons_*', 'keep *_particleFlow_photons_*', 'keep *_trackerDrivenElectronSeeds_preid_*', 'keep *_offlineBeamSpot_*_*', 'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtTriggerMenuLite_*_*', 'keep *_conditionsInEdm_*_*', 'keep *_l1extraParticles_*_*', 'keep *_l1L1GtObjectMap_*_*', 'keep L1MuGMTReadoutCollection_gtDigis_*_*', 'keep L1GctEmCand*_gctDigis_*_*', 'keep L1GctJetCand*_gctDigis_*_*', 'keep L1GctEtHad*_gctDigis_*_*', 'keep L1GctEtMiss*_gctDigis_*_*', 'keep L1GctEtTotal*_gctDigis_*_*', 'keep L1GctHtMiss*_gctDigis_*_*', 'keep L1GctJetCounts*_gctDigis_*_*', 'keep L1GctHFRingEtSums*_gctDigis_*_*', 'keep L1GctHFBitCounts*_gctDigis_*_*', 'keep LumiDetails_lumiProducer_*_*', 'keep LumiSummary_lumiProducer_*_*', 'drop *_hlt*_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep L1AcceptBunchCrossings_*_*_*', 'keep L1TriggerScalerss_*_*_*', 'keep Level1TriggerScalerss_*_*_*', 'keep LumiScalerss_*_*_*', 'keep BeamSpotOnlines_*_*_*', 'keep DcsStatuss_*_*_*', 'keep *_logErrorHarvester_*_*', 'drop *_MEtoEDMConverter_*_*', 'drop *_*_*_SKIM')),
    fileName = cms.untracked.string('ChiB.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('ChiB'),
        dataTier = cms.untracked.string('RECO')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880)
)

# Other statements
process.GlobalTag.globaltag = 'GR_R_52_V7::All'

# Path and EndPath definitions
process.SKIMStreamChiBOutPath = cms.EndPath(process.SKIMStreamChiB)

# Schedule definition
process.schedule = cms.Schedule(process.upsilonHLTPath,process.SKIMStreamChiBOutPath)

