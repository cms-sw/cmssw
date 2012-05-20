# Auto generated configuration file
# using: 
# Revision: 1.372.2.1 
# Source: /local/reps/CMSSW.admin/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: skims -s SKIM:@MultiJet --data --no_exec --dbs find file,file.parent where dataset=/MultiJet/Run2012A-PromptReco-v1/RECO and run=191277 -n 100 --conditions auto:com10 --python_filename=skim_MultiJet.py
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
    secondaryFileNames = cms.untracked.vstring('/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/2EB422D2-3687-E111-81D3-5404A6388698.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/B09F89E6-2A87-E111-A2DB-0025901D629C.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/A098AF28-2F87-E111-83FE-002481E0D958.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/F6F1613A-1B87-E111-96FA-001D09F2437B.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/A49B3901-4387-E111-BC71-003048D3C980.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/A0892CE5-4087-E111-8CD7-001D09F2305C.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/F61975FE-3987-E111-8C62-BCAEC518FF74.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/D64905A7-3787-E111-9F28-0030486780B4.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/4ABF977F-2987-E111-8891-001D09F248F8.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/7E719EA2-3C87-E111-B484-003048F24A04.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/96FF90C1-3E87-E111-BE80-003048F1C424.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/0A87A1D5-1E87-E111-9E42-003048F118D2.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/0C451E08-3287-E111-8128-002481E0D958.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/AC76FA95-4187-E111-BE78-003048F1BF66.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/D0D1507E-2987-E111-AAC9-5404A63886B1.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/E64C0F5E-1D87-E111-9174-0025901D62A6.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/E446CDDB-2087-E111-AD5D-5404A640A648.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/6A9E1998-2187-E111-A719-BCAEC518FF67.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/F28838E9-2587-E111-A795-003048D2BA82.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/EEEC7B85-2287-E111-933B-001D09F24399.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/1E137F0D-2887-E111-93C5-001D09F2447F.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/36EF3554-2787-E111-8AA0-001D09F2527B.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/DAF4A569-1687-E111-965C-BCAEC53296FB.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/340263A8-2687-E111-BB89-003048D2BB58.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/A41727AD-1987-E111-97FD-001D09F252DA.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/E61D317D-2487-E111-AF61-003048D37694.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/08D58B99-1A87-E111-9C7D-001D09F25267.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/0EA690C3-2D87-E111-A965-485B3962633D.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/54E08B69-1687-E111-AAA8-0025901D5D78.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/5E50D1CE-2387-E111-8BC0-0025901D627C.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/EA16D62A-3487-E111-875F-003048D3C982.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/DAC7802A-3487-E111-9FE1-0030486780B4.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/E0CEA1D5-1E87-E111-9272-003048F118E0.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/68031D0B-3E87-E111-A59B-003048D37580.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/583DB55F-2C87-E111-B528-0025B324400C.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/74D1391B-1987-E111-A17E-0025901D5E10.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/28778EEC-1687-E111-8063-001D09F2960F.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/22DFEB98-4687-E111-8B76-003048F024E0.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/3885475F-2C87-E111-A6CA-BCAEC5329713.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/901BF563-1D87-E111-855C-BCAEC5364C93.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/8889E026-3187-E111-A51E-00237DDBE41A.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/523F6C15-1E87-E111-AD58-0025901D62A6.root', 
        '/store/data/Run2012A/MultiJet/RAW/v1/000/191/277/C0DDDA36-3B87-E111-90A8-001D09F2A49C.root'),
    fileNames = cms.untracked.vstring('/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/FEA07D8D-E988-E111-9573-BCAEC53296F4.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/F6EEA0D3-F888-E111-89F8-003048D37580.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/E26023D0-FD88-E111-BE54-BCAEC5329701.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/DA214F2B-0989-E111-A5FC-001D09F248F8.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/D83216FC-E488-E111-9D6D-001D09F2A465.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/D665A46B-F788-E111-B3FA-003048D37694.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/D4092D90-E988-E111-A2E7-5404A640A63D.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/D09E38F4-E888-E111-B9AE-001D09F292D1.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/CE568AAC-FB88-E111-B794-003048D3C980.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/CE108436-0489-E111-B523-0030486780EC.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/CCA24C42-FF88-E111-A857-0030486780B4.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/C6EBABBB-ED88-E111-BEB6-E0CB4E5536AE.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/C2091F69-EA88-E111-95BF-001D09F253C0.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/C04084BC-ED88-E111-8789-5404A63886CE.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/BE2D7849-F588-E111-9493-001D09F23D1D.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/AC3F01B9-ED88-E111-A4A8-BCAEC5329703.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/A4B78A81-0389-E111-AE00-003048D37456.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/A4478968-FC88-E111-A01B-00215AEDFD74.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/9AF88AD7-F888-E111-B1EC-001D09F2A465.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/900BE4CF-FD88-E111-9C66-BCAEC532971C.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/8EACEE80-EE88-E111-8E11-0025901D5C88.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/828F4F51-FA88-E111-A65F-00237DDC5C24.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/7ED47F93-0A89-E111-9D70-0019B9F70468.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/7C592841-FF88-E111-8E91-001D09F29146.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/7A2AD59B-0589-E111-BBB8-003048D3C944.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/6E814DF7-FA88-E111-9C5E-003048D2BC30.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/6279CE19-FD88-E111-A54B-002481E0DEC6.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/60FECB53-0689-E111-8842-001D09F290CE.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/602E5E04-0C89-E111-B8F6-001D09F2B2CF.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/52F0B07F-EE88-E111-A70D-485B39897227.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/501620D6-EB88-E111-98DD-BCAEC518FF76.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/4A9D8351-FA88-E111-825B-002481E0D646.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/308E3558-0189-E111-ADD1-003048D3C982.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/28E350F4-E888-E111-AABB-001D09F29619.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/26849B32-F888-E111-9E5E-0025B32036E2.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/20F56A41-FF88-E111-BC46-003048D2BC30.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/2004888D-F088-E111-BFC1-5404A63886C0.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/1E5F77D3-F888-E111-B612-003048D2C16E.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/1C0B5B66-EC88-E111-B2E7-5404A63886AD.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/18C53A04-F088-E111-A285-0025901D5DB2.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/12E1CD63-EC88-E111-96E5-5404A63886B0.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/081AC680-EE88-E111-98FA-5404A6388698.root', 
        '/store/data/Run2012A/MultiJet/RECO/PromptReco-v1/000/191/277/04382DFB-E888-E111-85FD-002481E94C7E.root')
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
process.SKIMStreamHighMETOutPath = cms.EndPath(process.SKIMStreamHighMET)

# Schedule definition
process.schedule = cms.Schedule(process.pfPath,process.tcPath,process.SKIMStreamHighMETOutPath)

