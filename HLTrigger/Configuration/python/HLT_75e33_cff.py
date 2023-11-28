import FWCore.ParameterSet.Config as cms

fragment = cms.ProcessFragment( "HLT" )

### Non HLT-specific event-setups
fragment.load("CalibMuon/CSCCalibration/CSCChannelMapper_cfi")
fragment.load("CalibMuon/CSCCalibration/CSCIndexer_cfi")
fragment.load("RecoHGCal/TICL/tracksterSelectionTf_cfi")
fragment.load("RecoJets/Configuration/CaloTowersES_cfi")
fragment.load("RecoLocalCalo/EcalRecAlgos/EcalSeverityLevelESProducer_cfi")
fragment.load("RecoLocalCalo/HcalRecAlgos/hcalRecAlgoESProd_cfi")
fragment.load("RecoLocalCalo/HcalRecAlgos/hcalChannelPropertiesESProd_cfi")
fragment.load("RecoLocalTracker/Phase2TrackerRecHits/Phase2StripCPEESProducer_cfi")
fragment.load("RecoLocalTracker/SiPixelRecHits/PixelCPEGeneric_cfi")
fragment.load("RecoTracker/PixelTrackFitting/pixelTrackCleanerBySharedHits_cfi")
fragment.load("RecoTracker/PixelLowPtUtilities/ClusterShapeHitFilterESProducer_cfi")
fragment.load("RecoTracker/FinalTrackSelectors/trackAlgoPriorityOrder_cfi")
fragment.load("RecoTracker/MeasurementDet/MeasurementTrackerESProducer_cfi")
fragment.load("RecoTracker/TkNavigation/NavigationSchoolESProducer_cfi")
fragment.load("RecoTracker/TkSeedingLayers/TTRHBuilderWithoutAngle4PixelTriplets_cfi")
fragment.load("RecoTracker/TransientTrackingRecHit/TransientTrackingRecHitBuilder_cfi")
fragment.load("TrackPropagation/SteppingHelixPropagator/SteppingHelixPropagatorAny_cfi")
fragment.load("TrackingTools/GeomPropagators/AnyDirectionAnalyticalPropagator_cfi")
fragment.load("TrackingTools/GsfTracking/BwdAnalyticalPropagator_cfi")
fragment.load("TrackingTools/GsfTracking/CloseComponentsTSOSMerger_cfi")
fragment.load("TrackingTools/GsfTracking/FwdAnalyticalPropagator_cfi")
fragment.load("TrackingTools/GsfTracking/GsfElectronFittingSmoother_cfi")
fragment.load("TrackingTools/GsfTracking/GsfElectronMaterialEffects_cfi")
fragment.load("TrackingTools/GsfTracking/GsfElectronTrajectoryFitter_cfi")
fragment.load("TrackingTools/GsfTracking/GsfElectronTrajectorySmoother_cfi")
fragment.load("TrackingTools/GsfTracking/KullbackLeiblerTSOSDistance_cfi")
fragment.load("TrackingTools/GsfTracking/fwdGsfElectronPropagator_cff")
fragment.load("TrackingTools/KalmanUpdators/Chi2MeasurementEstimator_cfi")
fragment.load("TrackingTools/KalmanUpdators/KFUpdatorESProducer_cfi")
fragment.load("TrackingTools/MaterialEffects/MaterialPropagatorParabolicMf_cff")
fragment.load("TrackingTools/MaterialEffects/MaterialPropagator_cfi")
fragment.load("TrackingTools/MaterialEffects/OppositeMaterialPropagator_cfi")
fragment.load("TrackingTools/MaterialEffects/PropagatorsForLoopers_cff")
fragment.load("TrackingTools/MaterialEffects/RungeKuttaTrackerPropagator_cfi")
fragment.load("TrackingTools/RecoGeometry/GlobalDetLayerGeometryESProducer_cfi")
fragment.load("TrackingTools/TrackAssociator/DetIdAssociatorESProducer_cff")
fragment.load("TrackingTools/TrackFitters/FlexibleKFFittingSmoother_cfi")
fragment.load("TrackingTools/TrackFitters/LooperFitters_cff")
fragment.load("TrackingTools/TrackFitters/RungeKuttaFitters_cff")
fragment.load("TrackingTools/TrajectoryCleaning/TrajectoryCleanerBySharedHits_cfi")
fragment.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")

### Actual changes on top of Phase2
### It could come from RecoTracker/IterativeTracking/InitialStep_cff.py
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/initialStepChi2Est_cfi")
### It could come from RecoTracker/IterativeTracking/python/HighPtTripletStep_cff.py
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/highPtTripletStepChi2Est_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/highPtTripletStepTrajectoryCleanerBySharedHits_cfi")
### It could come from RecoTracker/IterativeTracking/python/MuonSeededStep_cff.py
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/muonSeededTrajectoryCleanerBySharedHits_cfi")

### Mostly comes from HLT-like configuration, not RECO-like configuration
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltOnlineBeamSpotESProducer_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPBwdElectronPropagator_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPChi2ChargeMeasurementEstimator2000_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPChi2ChargeMeasurementEstimator30_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPChi2MeasurementEstimator100_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPChi2MeasurementEstimator30_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPDummyDetLayerGeometry_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPFastSteppingHelixPropagatorAny_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPFastSteppingHelixPropagatorOpposite_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPFwdElectronPropagator_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPKFTrajectorySmootherForMuonTrackLoader_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPKFUpdator_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPL3MuKFTrajectoryFitter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPMuonTransientTrackingRecHitBuilder_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPRungeKuttaTrackerPropagator_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPSmartPropagator_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPSmartPropagatorAny_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPSmartPropagatorAnyOpposite_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPSteppingHelixPropagatorAlong_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPSteppingHelixPropagatorOpposite_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPTrackAlgoPriorityOrder_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPTrajectoryCleanerBySharedHits_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltPhase2L3MuonHighPtTripletStepChi2Est_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltPhase2L3MuonHighPtTripletStepTrajectoryCleanerBySharedHits_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltPhase2L3MuonInitialStepChi2Est_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltPhase2L3MuonPixelTrackCleanerBySharedHits_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltPhase2L3MuonTrackAlgoPriorityOrder_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltPixelTracksCleanerBySharedHits_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltTTRBWR_cfi")

fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPKFFittingSmootherForL2Muon_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPKFTrajectoryFitterForL2Muon_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPKFTrajectorySmootherForL2Muon_cfi")

fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/trackdnn_source_cfi")

fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_AK4PFPuppiJet520_cfi")

fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_DoubleMediumChargedIsoPFTauHPS40_eta2p1_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1_cfi")

fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Diphoton30_23_IsoCaloId_L1Seeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Diphoton30_23_IsoCaloId_Unseeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_DoubleEle23_12_Iso_L1Seeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_DoubleEle25_CaloIdL_PMS2_L1Seeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_DoubleEle25_CaloIdL_PMS2_Unseeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepCSV_2p4_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepFlavour_2p4_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Ele26_WP70_L1Seeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Ele26_WP70_Unseeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Ele115_NonIso_L1Seeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Ele32_WPTight_L1Seeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Ele32_WPTight_Unseeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_IsoMu24_FromL1TkMuon_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_FromL1TkMuon_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Mu37_Mu27_FromL1TkMuon_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Mu50_FromL1TkMuon_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepCSV_2p4_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepCSV_2p4_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_PFPuppiHT1070_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_PFPuppiMETTypeOne140_PFPuppiMHT140_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Photon108EB_TightID_TightIso_L1Seeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Photon108EB_TightID_TightIso_Unseeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Photon187_L1Seeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Photon187_Unseeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_TriMu_10_5_5_DZ_FromL1TkMuon_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_DoubleNNTau52_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_DoublePFPuppiJets112_2p4_DEta1p6_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_DoubleTkMuon_15_7_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_PFHT400PT30_QuadPFPuppiJet_70_55_40_40_2p4_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_PFPuppiHT450off_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_PFPuppiMET220off_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_SingleNNTau150_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_SinglePFPuppiJet230off_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_SingleTkMuon_22_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_TkEle25TkEle12_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_TkEle36_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_TkEm37TkEm24_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_TkEm51_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_TkIsoEle22TkEm12_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_TkIsoEle28_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_TkIsoEm22TkIsoEm12_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_TkIsoEm36_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/L1T_TripleTkMuon_5_3_3_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/MC_BTV_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/MC_Ele5_Open_L1Seeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/MC_Ele5_Open_Unseeded_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/MC_JME_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/CkfBaseTrajectoryFilter_block_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/ckfBaseTrajectoryFilterP5_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/CkfTrajectoryBuilder_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/ckfTrajectoryFilterBeamHaloMuon_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/ClusterShapeTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/conv2CkfTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/convCkfTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/CSCSegAlgoDF_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/CSCSegAlgoRU_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/CSCSegAlgoSK_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/CSCSegAlgoST_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/CSCSegAlgoTC_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/detachedQuadStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/detachedQuadStepTrajectoryFilterBase_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/detachedTripletStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/detachedTripletStepTrajectoryFilterBase_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/DTLinearDriftFromDBAlgo_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/GlobalMuonTrackMatcher_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/GroupedCkfTrajectoryBuilder_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HFRecalParameterBlock_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HGCAL_cceParams_toUse_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HGCAL_chargeCollectionEfficiencies_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HGCAL_ileakParam_toUse_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HGCAL_noise_fC_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HGCAL_noise_heback_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hgceeDigitizer_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hgchebackDigitizer_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hgchefrontDigitizer_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/highPtTripletStepTrajectoryBuilder_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/highPtTripletStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/highPtTripletStepTrajectoryFilterBase_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/highPtTripletStepTrajectoryFilterInOut_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HLTIter0Phase2L3FromL1TkMuonGroupedCkfTrajectoryFilterIT_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HLTIter0Phase2L3FromL1TkMuonPSetGroupedCkfTrajectoryBuilderIT_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HLTIter2Phase2L3FromL1TkMuonPSetGroupedCkfTrajectoryBuilderIT_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HLTIter2Phase2L3FromL1TkMuonPSetTrajectoryFilterIT_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hltPhase2L3MuonHighPtTripletStepTrajectoryBuilder_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hltPhase2L3MuonHighPtTripletStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hltPhase2L3MuonHighPtTripletStepTrajectoryFilterBase_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hltPhase2L3MuonHighPtTripletStepTrajectoryFilterInOut_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hltPhase2L3MuonInitialStepTrajectoryBuilder_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hltPhase2L3MuonInitialStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hltPhase2L3MuonPSetPvClusterComparerForIT_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hltPhase2L3MuonSeedFromProtoTracks_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hltPhase2PSetPvClusterComparerForIT_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/hltPhase2SeedFromProtoTracks_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HLTPSetMuonCkfTrajectoryBuilder_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HLTPSetMuonCkfTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HLTPSetTrajectoryBuilderForGsfElectrons_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HLTPSetTrajectoryFilterForElectrons_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HLTSiStripClusterChargeCutLoose_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/HLTSiStripClusterChargeCutNone_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/initialStepTrajectoryBuilder_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/initialStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/initialStepTrajectoryFilterBasePreSplitting_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/initialStepTrajectoryFilterPreSplitting_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/initialStepTrajectoryFilterShapePreSplitting_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/jetCoreRegionalStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/lowPtGsfEleTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/lowPtQuadStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/lowPtQuadStepTrajectoryFilterBase_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/lowPtTripletStepStandardTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/lowPtTripletStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/lowPtTripletStepTrajectoryFilterInOut_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/ME0SegAlgoRU_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/ME0SegmentAlgorithm_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/mixedTripletStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/muonSeededTrajectoryBuilderForInOut_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/muonSeededTrajectoryBuilderForOutInDisplaced_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/muonSeededTrajectoryFilterForInOut_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/muonSeededTrajectoryFilterForOutIn_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/muonSeededTrajectoryFilterForOutInDisplaced_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/pixelLessStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/pixelPairStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/pixelPairStepTrajectoryFilterBase_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/pixelPairStepTrajectoryFilterInOut_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/PixelTripletHLTGenerator_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/pSetPvClusterComparerForIT_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/seedFromProtoTracks_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/SiStripClusterChargeCutLoose_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/SiStripClusterChargeCutNone_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/SiStripClusterChargeCutTight_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/tobTecStepInOutTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/tobTecStepTrajectoryFilter_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/TrajectoryFilterForConversions_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/psets/TrajectoryFilterForElectrons_cfi")

fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLTriggerFinalPath_cff")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLTAnalyzerEndpath_cff")

fragment.schedule = cms.Schedule(*[

    fragment.L1T_SinglePFPuppiJet230off,
    fragment.L1T_PFPuppiHT450off,
    fragment.L1T_PFPuppiMET220off,

    fragment.HLT_AK4PFPuppiJet520,
    fragment.HLT_PFPuppiHT1070,
    fragment.HLT_PFPuppiMETTypeOne140_PFPuppiMHT140,

    fragment.L1T_PFHT400PT30_QuadPFPuppiJet_70_55_40_40_2p4,
    fragment.L1T_DoublePFPuppiJets112_2p4_DEta1p6,

    fragment.HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepCSV_2p4,
    fragment.HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepCSV_2p4,
    fragment.HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepCSV_2p4,
    fragment.HLT_PFHT330PT30_QuadPFPuppiJet_75_60_45_40_TriplePFPuppiBTagDeepFlavour_2p4,
    fragment.HLT_PFHT200PT30_QuadPFPuppiJet_70_40_30_30_TriplePFPuppiBTagDeepFlavour_2p4,
    fragment.HLT_DoublePFPuppiJets128_DoublePFPuppiBTagDeepFlavour_2p4,

    fragment.L1T_SingleTkMuon_22,
    fragment.L1T_DoubleTkMuon_15_7,
    fragment.L1T_TripleTkMuon_5_3_3,

    fragment.HLT_Mu50_FromL1TkMuon,
    fragment.HLT_IsoMu24_FromL1TkMuon,
    fragment.HLT_Mu37_Mu27_FromL1TkMuon,
    fragment.HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_FromL1TkMuon,
    fragment.HLT_TriMu_10_5_5_DZ_FromL1TkMuon,

    fragment.L1T_TkEm51,
    fragment.L1T_TkEle36,
    fragment.L1T_TkIsoEm36,
    fragment.L1T_TkIsoEle28,
    fragment.L1T_TkEm37TkEm24,
    fragment.L1T_TkEle25TkEle12,
    fragment.L1T_TkIsoEm22TkIsoEm12,
    fragment.L1T_TkIsoEle22TkEm12,

    fragment.HLT_Ele32_WPTight_Unseeded,
    fragment.HLT_Ele26_WP70_Unseeded,
    fragment.HLT_Photon108EB_TightID_TightIso_Unseeded,
    fragment.HLT_Photon187_Unseeded,
    fragment.HLT_DoubleEle25_CaloIdL_PMS2_Unseeded,
    fragment.HLT_Diphoton30_23_IsoCaloId_Unseeded,
    fragment.HLT_Ele32_WPTight_L1Seeded,
    fragment.HLT_Ele115_NonIso_L1Seeded,
    fragment.HLT_Ele26_WP70_L1Seeded,
    fragment.HLT_Photon108EB_TightID_TightIso_L1Seeded,
    fragment.HLT_Photon187_L1Seeded,
    fragment.HLT_DoubleEle25_CaloIdL_PMS2_L1Seeded,
    fragment.HLT_DoubleEle23_12_Iso_L1Seeded,
    fragment.HLT_Diphoton30_23_IsoCaloId_L1Seeded,

    fragment.HLT_DoubleMediumChargedIsoPFTauHPS40_eta2p1,
    ### Removed temporarily until solution of https://github.com/cms-sw/cmssw/issues/42862
    #fragment.HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1,

    ### Removed temporarily until final decision on L1T tau Phase-2
    #fragment.L1T_DoubleNNTau52,
    #fragment.L1T_SingleNNTau150,

    fragment.MC_JME,
    fragment.MC_BTV,
    fragment.MC_Ele5_Open_Unseeded,
    fragment.MC_Ele5_Open_L1Seeded,

    fragment.HLTriggerFinalPath,
    fragment.HLTAnalyzerEndpath,
])
