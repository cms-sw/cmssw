# Auto generated configuration file
# using: 
# Revision: 1.372.2.1 
# Source: /local/reps/CMSSW.admin/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: skims -s SKIM:@Photon --data --no_exec --dbs find file,file.parent where dataset=/Photon/Run2012A-PromptReco-v1/RECO and run=191277 -n 100 --conditions auto:com10 --python_filename=skim_Photon.py
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
    secondaryFileNames = cms.untracked.vstring('/store/data/Run2012A/Photon/RAW/v1/000/191/277/3C952B20-1987-E111-8D29-485B3977172C.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/E2495EF1-3787-E111-8F8F-001D09F241B9.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/82ADEBCE-2387-E111-8FFA-0025901D625A.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/6681DAA9-1A87-E111-99DB-001D09F25479.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/B88A51CE-1E87-E111-97D1-5404A63886AE.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/56306F8D-3587-E111-9698-BCAEC518FF30.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/30E64CC8-2887-E111-9AAB-003048D37538.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/FCF6E72E-4087-E111-B961-003048D374F2.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/D2D6EE86-3A87-E111-8140-003048F1C424.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/6E1B81FE-3187-E111-BEEE-003048F11942.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/1AECD63D-2587-E111-8EB2-0030486730C6.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/5E12E054-2C87-E111-82E3-0025901D626C.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/32088FC1-2D87-E111-874E-003048D37524.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/ECFB17E3-2F87-E111-9A80-003048F118C4.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/0621095E-1687-E111-99A0-003048F024F6.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/9CB3E72E-2A87-E111-A334-003048D37456.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/1A6408EE-1B87-E111-B8A4-00215AEDFCCC.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/FE76B510-1E87-E111-B2A1-003048D2C01A.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/92F5108E-4687-E111-AA59-001D09F290BF.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/461AF455-2787-E111-9AEF-003048D37456.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/E6422683-2287-E111-9503-0025901D5D90.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/F4C59496-4187-E111-A8B9-001D09F25041.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/FA02E209-3E87-E111-806A-001D09F24FBA.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/304978EC-1687-E111-8E4B-001D09F2424A.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/507E9258-1D87-E111-9F99-002481E0D958.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/0AA572A7-2687-E111-A985-003048D2BEA8.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/68099397-2187-E111-AE37-BCAEC518FF68.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/8E00A59A-2B87-E111-ADE7-001D09F276CF.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/0E9ED700-4387-E111-A3E3-003048D37456.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/1E28ADED-3B87-E111-A3C2-003048F11942.root', 
        '/store/data/Run2012A/Photon/RAW/v1/000/191/277/7E0B9920-3487-E111-BFB3-BCAEC518FF30.root'),
    fileNames = cms.untracked.vstring('/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/EA68E010-EB88-E111-848C-0019B9F70468.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/DA8941E3-F688-E111-96FD-0025901D5E10.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/D6F3DA02-0089-E111-96C0-001D09F2932B.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/C28EC5FF-FF88-E111-954B-0019B9F72F97.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/C0FDEFF2-E888-E111-A013-003048D2BE12.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/BEFA68F9-FA88-E111-96AA-003048D37456.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/B83A05D5-F888-E111-B9EC-0025B32036E2.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/AA4EEC24-0289-E111-BF20-001D09F28D54.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/A0707845-F288-E111-9100-0025901D5C88.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/9C591435-0489-E111-82DB-001D09F24303.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/96D68107-0789-E111-9502-003048D37580.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/8A901316-EB88-E111-B92A-0025B32036D2.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/86508696-F088-E111-BEE7-5404A638869E.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/7C206C8D-E988-E111-AB16-5404A63886A2.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/72B71F44-ED88-E111-93FD-003048D2BE12.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/6EFDB096-F988-E111-A78A-003048D2BC42.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/6E9CBABD-ED88-E111-BE07-00237DDBE49C.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/6CFF2581-0389-E111-A803-003048D2BD66.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/62A79794-FE88-E111-8F25-003048D2BBF0.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/5CCB5060-0189-E111-ACE2-001D09F25041.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/5C4B9099-FE88-E111-B222-003048D2BC30.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/5AA0DDE2-F688-E111-B406-E0CB4E55365D.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/5882B65C-F688-E111-98B3-BCAEC518FF8F.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/54CA1A51-FA88-E111-B867-0025B32035BC.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/3E957ACA-0289-E111-B7D9-003048D2BBF0.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/2CB42C50-FA88-E111-BAC8-0025B32034EA.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/2AF7FD51-EB88-E111-B96E-001D09F23F2A.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/1A72CC58-EC88-E111-90E8-485B39897227.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/14C3BA58-F488-E111-9E4D-5404A63886CC.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/12CBFB21-0289-E111-A007-003048D374CA.root', 
        '/store/data/Run2012A/Photon/RECO/PromptReco-v1/000/191/277/04FA672F-EF88-E111-8857-BCAEC5329703.root')
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

