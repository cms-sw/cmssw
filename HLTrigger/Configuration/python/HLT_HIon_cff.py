# /dev/CMSSW_4_4_2/HIon/V41 (CMSSW_4_4_0_HLT17)

import FWCore.ParameterSet.Config as cms


HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_4_4_2/HIon/V41')
)

streams = cms.PSet( 
  A = cms.vstring( 'HIDiMuon',
    'HIHighPt',
    'HIMinBiasUPC' ),
  Calibration = cms.vstring( 'TestEnablesEcalHcalDT' ),
  DQM = cms.vstring( 'OnlineMonitorHI' ),
  EcalCalibration = cms.vstring( 'EcalLaser' ),
  HIExpress = cms.vstring( 'HIExpressPhysics' ),
  HLTDQM = cms.vstring( 'OnlineHltMonitorHI' ),
  HLTDQMResults = cms.vstring( 'OnlineHltResults' ),
  HLTMON = cms.vstring( 'OfflineMonitorHI' )
)
datasets = cms.PSet( 
  EcalLaser = cms.vstring( 'HLT_HIEcalCalibration_v1' ),
  HIDiMuon = cms.vstring( 'HLT_HIL1DoubleMu0_HighQ_v2',
    'HLT_HIL1DoubleMuOpen_v2',
    'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v2',
    'HLT_HIL2DoubleMu0_NHitQ_v2',
    'HLT_HIL2DoubleMu0_v2',
    'HLT_HIL2DoubleMu3_v2',
    'HLT_HIL2Mu15_v2',
    'HLT_HIL2Mu3_NHitQ_v2',
    'HLT_HIL2Mu3_v2',
    'HLT_HIL2Mu7_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_SS_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_v2',
    'HLT_HIL3DoubleMuOpen_v2',
    'HLT_HIL3Mu3_v2' ),
  HIExpressPhysics = cms.vstring( 'HLT_HIBptxXOR_v1',
    'HLT_HICentral10_v2',
    'HLT_HICentralityVeto_v1',
    'HLT_HIDiJet55_v1',
    'HLT_HIDoublePhoton10_v1',
    'HLT_HIDoublePhoton15_v1',
    'HLT_HIDoublePhoton20_v1',
    'HLT_HIFullTrack12_L1Central_v1',
    'HLT_HIFullTrack12_L1Peripheral_v1',
    'HLT_HIFullTrack14_L1Central_v1',
    'HLT_HIFullTrack14_L1Peripheral_v1',
    'HLT_HIFullTrack20_L1Central_v1',
    'HLT_HIFullTrack20_L1Peripheral_v1',
    'HLT_HIFullTrack25_L1Central_v1',
    'HLT_HIFullTrack25_L1Peripheral_v1',
    'HLT_HIJet55_v1',
    'HLT_HIJet65_Jet55_v1',
    'HLT_HIJet65_v1',
    'HLT_HIJet80_v1',
    'HLT_HIJet95_v1',
    'HLT_HIJetE30_NoBPTX_v1',
    'HLT_HIJetE50_NoBPTX3BX_NoHalo_v1',
    'HLT_HIL1Algo_BptxXOR_BSC_OR_v1',
    'HLT_HIL1DoubleMu0_HighQ_v2',
    'HLT_HIL1DoubleMuOpen_v2',
    'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v2',
    'HLT_HIL2DoubleMu0_NHitQ_v2',
    'HLT_HIL2DoubleMu0_v2',
    'HLT_HIL2DoubleMu3_v2',
    'HLT_HIL2Mu15_v2',
    'HLT_HIL2Mu3_NHitQ_v2',
    'HLT_HIL2Mu3_v2',
    'HLT_HIL2Mu7_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_SS_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_v2',
    'HLT_HIL3DoubleMuOpen_v2',
    'HLT_HIL3Mu3_v2',
    'HLT_HIMET120_v1',
    'HLT_HIMET200_v1',
    'HLT_HIMET220_v1',
    'HLT_HIMinBiasBSC_OR_v1',
    'HLT_HIMinBiasBSC_v1',
    'HLT_HIMinBiasHF_v1',
    'HLT_HIMinBiasHfOrBSC_v1',
    'HLT_HIMinBiasHf_OR_v1',
    'HLT_HIMinBiasPixel_SingleTrack_v1',
    'HLT_HIMinBiasZDCPixel_SingleTrack_v1',
    'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v1',
    'HLT_HIMinBiasZDC_Calo_v1',
    'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v1',
    'HLT_HIPhoton10_Photon15_v1',
    'HLT_HIPhoton15_Photon20_v1',
    'HLT_HIPhysics_v1',
    'HLT_HIRandom_v1',
    'HLT_HISinglePhoton15_v1',
    'HLT_HISinglePhoton20_v2',
    'HLT_HISinglePhoton30_v2',
    'HLT_HISinglePhoton40_v2',
    'HLT_HIUCC010_v2',
    'HLT_HIUCC015_v2',
    'HLT_HIUPCNeuEG2Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuEG5Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v1',
    'HLT_HIUPCNeuMuPixel_SingleTrack_v1',
    'HLT_HIZeroBiasPixel_SingleTrack_v1',
    'HLT_HIZeroBiasXOR_v1',
    'HLT_HIZeroBias_v1' ),
  HIHighPt = cms.vstring( 'HLT_HIDiJet55_v1',
    'HLT_HIDoublePhoton10_v1',
    'HLT_HIDoublePhoton15_v1',
    'HLT_HIDoublePhoton20_v1',
    'HLT_HIFullTrack12_L1Central_v1',
    'HLT_HIFullTrack12_L1Peripheral_v1',
    'HLT_HIFullTrack14_L1Central_v1',
    'HLT_HIFullTrack14_L1Peripheral_v1',
    'HLT_HIFullTrack20_L1Central_v1',
    'HLT_HIFullTrack20_L1Peripheral_v1',
    'HLT_HIFullTrack25_L1Central_v1',
    'HLT_HIFullTrack25_L1Peripheral_v1',
    'HLT_HIJet55_v1',
    'HLT_HIJet65_Jet55_v1',
    'HLT_HIJet65_v1',
    'HLT_HIJet80_v1',
    'HLT_HIJet95_v1',
    'HLT_HIJetE30_NoBPTX_v1',
    'HLT_HIJetE50_NoBPTX3BX_NoHalo_v1',
    'HLT_HIMET120_v1',
    'HLT_HIMET200_v1',
    'HLT_HIMET220_v1',
    'HLT_HIPhoton10_Photon15_v1',
    'HLT_HIPhoton15_Photon20_v1',
    'HLT_HISinglePhoton15_v1',
    'HLT_HISinglePhoton20_v2',
    'HLT_HISinglePhoton30_v2',
    'HLT_HISinglePhoton40_v2' ),
  HIMinBiasUPC = cms.vstring( 'HLT_HIBptxXOR_v1',
    'HLT_HICentral10_v2',
    'HLT_HICentralityVeto_v1',
    'HLT_HIL1Algo_BptxXOR_BSC_OR_v1',
    'HLT_HIMinBiasBSC_OR_v1',
    'HLT_HIMinBiasBSC_v1',
    'HLT_HIMinBiasHF_v1',
    'HLT_HIMinBiasHfOrBSC_v1',
    'HLT_HIMinBiasHf_OR_v1',
    'HLT_HIMinBiasPixel_SingleTrack_v1',
    'HLT_HIMinBiasZDCPixel_SingleTrack_v1',
    'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v1',
    'HLT_HIMinBiasZDC_Calo_v1',
    'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v1',
    'HLT_HIPhysics_v1',
    'HLT_HIRandom_v1',
    'HLT_HIUCC010_v2',
    'HLT_HIUCC015_v2',
    'HLT_HIUPCNeuEG2Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuEG5Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v1',
    'HLT_HIUPCNeuMuPixel_SingleTrack_v1',
    'HLT_HIZeroBiasPixel_SingleTrack_v1',
    'HLT_HIZeroBiasXOR_v1',
    'HLT_HIZeroBias_v1' ),
  OfflineMonitorHI = cms.vstring( 'HLT_HIActivityHF_Coincidence3_v1',
    'HLT_HIActivityHF_Single3_v1',
    'HLT_HIBptxXOR_v1',
    'HLT_HICentral10_v2',
    'HLT_HICentralityVeto_v1',
    'HLT_HIClusterVertexCompatibility_v1',
    'HLT_HIDiJet55_v1',
    'HLT_HIDoublePhoton10_v1',
    'HLT_HIDoublePhoton15_v1',
    'HLT_HIDoublePhoton20_v1',
    'HLT_HIFullTrack12_L1Central_v1',
    'HLT_HIFullTrack12_L1Peripheral_v1',
    'HLT_HIFullTrack14_L1Central_v1',
    'HLT_HIFullTrack14_L1Peripheral_v1',
    'HLT_HIFullTrack20_L1Central_v1',
    'HLT_HIFullTrack20_L1Peripheral_v1',
    'HLT_HIFullTrack25_L1Central_v1',
    'HLT_HIFullTrack25_L1Peripheral_v1',
    'HLT_HIJet55_v1',
    'HLT_HIJet65_Jet55_v1',
    'HLT_HIJet65_v1',
    'HLT_HIJet80_v1',
    'HLT_HIJet95_v1',
    'HLT_HIJetE30_NoBPTX_v1',
    'HLT_HIJetE50_NoBPTX3BX_NoHalo_v1',
    'HLT_HIL1Algo_BptxXOR_BSC_OR_v1',
    'HLT_HIL1DoubleMu0_HighQ_v2',
    'HLT_HIL1DoubleMuOpen_v2',
    'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v2',
    'HLT_HIL2DoubleMu0_NHitQ_v2',
    'HLT_HIL2DoubleMu0_v2',
    'HLT_HIL2DoubleMu3_v2',
    'HLT_HIL2Mu15_v2',
    'HLT_HIL2Mu3_NHitQ_v2',
    'HLT_HIL2Mu3_v2',
    'HLT_HIL2Mu7_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_SS_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_v2',
    'HLT_HIL3DoubleMuOpen_v2',
    'HLT_HIL3Mu3_v2',
    'HLT_HIMET120_v1',
    'HLT_HIMET200_v1',
    'HLT_HIMET220_v1',
    'HLT_HIMinBiasBSC_OR_v1',
    'HLT_HIMinBiasBSC_v1',
    'HLT_HIMinBiasHF_v1',
    'HLT_HIMinBiasHfOrBSC_v1',
    'HLT_HIMinBiasHf_OR_v1',
    'HLT_HIMinBiasPixel_SingleTrack_v1',
    'HLT_HIMinBiasZDCPixel_SingleTrack_v1',
    'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v1',
    'HLT_HIMinBiasZDC_Calo_v1',
    'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v1',
    'HLT_HIPhoton10_Photon15_v1',
    'HLT_HIPhoton15_Photon20_v1',
    'HLT_HIPhysics_v1',
    'HLT_HIRandom_v1',
    'HLT_HISinglePhoton15_v1',
    'HLT_HISinglePhoton20_v2',
    'HLT_HISinglePhoton30_v2',
    'HLT_HISinglePhoton40_v2',
    'HLT_HIUCC010_v2',
    'HLT_HIUCC015_v2',
    'HLT_HIUPCNeuEG2Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuEG5Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v1',
    'HLT_HIUPCNeuMuPixel_SingleTrack_v1',
    'HLT_HIZeroBiasPixel_SingleTrack_v1',
    'HLT_HIZeroBiasXOR_v1',
    'HLT_HIZeroBias_v1' ),
  OnlineHltMonitorHI = cms.vstring( 'HLT_HIActivityHF_Coincidence3_v1',
    'HLT_HIActivityHF_Single3_v1',
    'HLT_HIBptxXOR_v1',
    'HLT_HICentral10_v2',
    'HLT_HICentralityVeto_v1',
    'HLT_HIClusterVertexCompatibility_v1',
    'HLT_HIDiJet55_v1',
    'HLT_HIDoublePhoton10_v1',
    'HLT_HIDoublePhoton15_v1',
    'HLT_HIDoublePhoton20_v1',
    'HLT_HIFullTrack12_L1Central_v1',
    'HLT_HIFullTrack12_L1Peripheral_v1',
    'HLT_HIFullTrack14_L1Central_v1',
    'HLT_HIFullTrack14_L1Peripheral_v1',
    'HLT_HIFullTrack20_L1Central_v1',
    'HLT_HIFullTrack20_L1Peripheral_v1',
    'HLT_HIFullTrack25_L1Central_v1',
    'HLT_HIFullTrack25_L1Peripheral_v1',
    'HLT_HIJet55_v1',
    'HLT_HIJet65_Jet55_v1',
    'HLT_HIJet65_v1',
    'HLT_HIJet80_v1',
    'HLT_HIJet95_v1',
    'HLT_HIJetE30_NoBPTX_v1',
    'HLT_HIJetE50_NoBPTX3BX_NoHalo_v1',
    'HLT_HIL1Algo_BptxXOR_BSC_OR_v1',
    'HLT_HIL1DoubleMu0_HighQ_v2',
    'HLT_HIL1DoubleMuOpen_v2',
    'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v2',
    'HLT_HIL2DoubleMu0_NHitQ_v2',
    'HLT_HIL2DoubleMu0_v2',
    'HLT_HIL2DoubleMu3_v2',
    'HLT_HIL2Mu15_v2',
    'HLT_HIL2Mu3_NHitQ_v2',
    'HLT_HIL2Mu3_v2',
    'HLT_HIL2Mu7_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_SS_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_v2',
    'HLT_HIL3DoubleMuOpen_v2',
    'HLT_HIL3Mu3_v2',
    'HLT_HIMET120_v1',
    'HLT_HIMET200_v1',
    'HLT_HIMET220_v1',
    'HLT_HIMinBiasBSC_OR_v1',
    'HLT_HIMinBiasBSC_v1',
    'HLT_HIMinBiasHF_v1',
    'HLT_HIMinBiasHfOrBSC_v1',
    'HLT_HIMinBiasHf_OR_v1',
    'HLT_HIMinBiasPixel_SingleTrack_v1',
    'HLT_HIMinBiasZDCPixel_SingleTrack_v1',
    'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v1',
    'HLT_HIMinBiasZDC_Calo_v1',
    'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v1',
    'HLT_HIPhoton10_Photon15_v1',
    'HLT_HIPhoton15_Photon20_v1',
    'HLT_HIPhysics_v1',
    'HLT_HIRandom_v1',
    'HLT_HISinglePhoton15_v1',
    'HLT_HISinglePhoton20_v2',
    'HLT_HISinglePhoton30_v2',
    'HLT_HISinglePhoton40_v2',
    'HLT_HIUCC010_v2',
    'HLT_HIUCC015_v2',
    'HLT_HIUPCNeuEG2Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuEG5Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v1',
    'HLT_HIUPCNeuMuPixel_SingleTrack_v1',
    'HLT_HIZeroBiasPixel_SingleTrack_v1',
    'HLT_HIZeroBiasXOR_v1',
    'HLT_HIZeroBias_v1' ),
  OnlineHltResults = cms.vstring( 'HLTriggerFinalPath' ),
  OnlineMonitorHI = cms.vstring( 'HLT_HIActivityHF_Coincidence3_v1',
    'HLT_HIActivityHF_Single3_v1',
    'HLT_HIBptxXOR_v1',
    'HLT_HICentral10_v2',
    'HLT_HICentralityVeto_v1',
    'HLT_HIClusterVertexCompatibility_v1',
    'HLT_HIDTCalibration_v1',
    'HLT_HIDiJet55_v1',
    'HLT_HIDoublePhoton10_v1',
    'HLT_HIDoublePhoton15_v1',
    'HLT_HIDoublePhoton20_v1',
    'HLT_HIEcalCalibration_v1',
    'HLT_HIFullTrack12_L1Central_v1',
    'HLT_HIFullTrack12_L1Peripheral_v1',
    'HLT_HIFullTrack14_L1Central_v1',
    'HLT_HIFullTrack14_L1Peripheral_v1',
    'HLT_HIFullTrack20_L1Central_v1',
    'HLT_HIFullTrack20_L1Peripheral_v1',
    'HLT_HIFullTrack25_L1Central_v1',
    'HLT_HIFullTrack25_L1Peripheral_v1',
    'HLT_HIHcalCalibration_v1',
    'HLT_HIJet55_v1',
    'HLT_HIJet65_Jet55_v1',
    'HLT_HIJet65_v1',
    'HLT_HIJet80_v1',
    'HLT_HIJet95_v1',
    'HLT_HIJetE30_NoBPTX_v1',
    'HLT_HIJetE50_NoBPTX3BX_NoHalo_v1',
    'HLT_HIL1Algo_BptxXOR_BSC_OR_v1',
    'HLT_HIL1DoubleMu0_HighQ_v2',
    'HLT_HIL1DoubleMuOpen_v2',
    'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v2',
    'HLT_HIL2DoubleMu0_NHitQ_v2',
    'HLT_HIL2DoubleMu0_v2',
    'HLT_HIL2DoubleMu3_v2',
    'HLT_HIL2Mu15_v2',
    'HLT_HIL2Mu3_NHitQ_v2',
    'HLT_HIL2Mu3_v2',
    'HLT_HIL2Mu7_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_SS_v2',
    'HLT_HIL3DoubleMuOpen_Mgt2_v2',
    'HLT_HIL3DoubleMuOpen_v2',
    'HLT_HIL3Mu3_v2',
    'HLT_HIMET120_v1',
    'HLT_HIMET200_v1',
    'HLT_HIMET220_v1',
    'HLT_HIMinBiasBSC_OR_v1',
    'HLT_HIMinBiasBSC_v1',
    'HLT_HIMinBiasHF_v1',
    'HLT_HIMinBiasHfOrBSC_v1',
    'HLT_HIMinBiasHf_OR_v1',
    'HLT_HIMinBiasPixel_SingleTrack_v1',
    'HLT_HIMinBiasZDCPixel_SingleTrack_v1',
    'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v1',
    'HLT_HIMinBiasZDC_Calo_v1',
    'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v1',
    'HLT_HIPhoton10_Photon15_v1',
    'HLT_HIPhoton15_Photon20_v1',
    'HLT_HIPhysics_v1',
    'HLT_HIRandom_v1',
    'HLT_HISinglePhoton15_v1',
    'HLT_HISinglePhoton20_v2',
    'HLT_HISinglePhoton30_v2',
    'HLT_HISinglePhoton40_v2',
    'HLT_HIUCC010_v2',
    'HLT_HIUCC015_v2',
    'HLT_HIUPCNeuEG2Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuEG5Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v1',
    'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v1',
    'HLT_HIUPCNeuMuPixel_SingleTrack_v1',
    'HLT_HIZeroBiasPixel_SingleTrack_v1',
    'HLT_HIZeroBiasXOR_v1',
    'HLT_HIZeroBias_v1' ),
  TestEnablesEcalHcalDT = cms.vstring( 'HLT_HIDTCalibration_v1',
    'HLT_HIEcalCalibration_v1',
    'HLT_HIHcalCalibration_v1' )
)

hltESSAK5CaloL1L2L3 = cms.ESSource( "JetCorrectionServiceChain",
  appendToDataLabel = cms.string( "" ),
  correctors = cms.vstring( 'hltESSL1FastJetCorrectionService',
    'hltESSL2RelativeCorrectionService',
    'hltESSL3AbsoluteCorrectionService' ),
  label = cms.string( "hltESSAK5CaloL1L2L3" )
)
hltESSAK5CaloL2L3 = cms.ESSource( "JetCorrectionServiceChain",
  appendToDataLabel = cms.string( "" ),
  correctors = cms.vstring( 'hltESSL2RelativeCorrectionService',
    'hltESSL3AbsoluteCorrectionService' ),
  label = cms.string( "hltESSAK5CaloL2L3" )
)
hltESSBTagRecord = cms.ESSource( "EmptyESSource",
  recordName = cms.string( "JetTagComputerRecord" ),
  iovIsRunNotTime = cms.bool( True ),
  appendToDataLabel = cms.string( "" ),
  firstValid = cms.vuint32( 1 )
)
hltESSEcalSeverityLevel = cms.ESSource( "EmptyESSource",
  recordName = cms.string( "EcalSeverityLevelAlgoRcd" ),
  iovIsRunNotTime = cms.bool( True ),
  appendToDataLabel = cms.string( "" ),
  firstValid = cms.vuint32( 1 )
)
hltESSHcalSeverityLevel = cms.ESSource( "EmptyESSource",
  recordName = cms.string( "HcalSeverityLevelComputerRcd" ),
  iovIsRunNotTime = cms.bool( True ),
  appendToDataLabel = cms.string( "" ),
  firstValid = cms.vuint32( 1 )
)
hltESSL1FastJetCorrectionService = cms.ESSource( "L1FastjetCorrectionService",
  appendToDataLabel = cms.string( "" ),
  era = cms.string( "Jec10V1" ),
  level = cms.string( "L1FastJet" ),
  algorithm = cms.string( "AK5Calo" ),
  section = cms.string( "" ),
  srcRho = cms.InputTag( 'hltKT6CaloJets','rho' ),
  useCondDB = cms.untracked.bool( True )
)
hltESSL2RelativeCorrectionService = cms.ESSource( "LXXXCorrectionService",
  appendToDataLabel = cms.string( "" ),
  level = cms.string( "L2Relative" ),
  algorithm = cms.string( "AK5Calo" ),
  section = cms.string( "" ),
  era = cms.string( "" ),
  useCondDB = cms.untracked.bool( True )
)
hltESSL3AbsoluteCorrectionService = cms.ESSource( "LXXXCorrectionService",
  appendToDataLabel = cms.string( "" ),
  level = cms.string( "L3Absolute" ),
  algorithm = cms.string( "AK5Calo" ),
  section = cms.string( "" ),
  era = cms.string( "" ),
  useCondDB = cms.untracked.bool( True )
)

AnyDirectionAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  ComponentName = cms.string( "AnyDirectionAnalyticalPropagator" ),
  PropagationDirection = cms.string( "anyDirection" ),
  MaxDPhi = cms.double( 1.6 ),
  appendToDataLabel = cms.string( "" )
)
CaloTopologyBuilder = cms.ESProducer( "CaloTopologyBuilder",
  appendToDataLabel = cms.string( "" )
)
CaloTowerConstituentsMapBuilder = cms.ESProducer( "CaloTowerConstituentsMapBuilder",
  MapFile = cms.untracked.string( "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz" ),
  appendToDataLabel = cms.string( "" )
)
CastorDbProducer = cms.ESProducer( "CastorDbProducer",
  appendToDataLabel = cms.string( "" )
)
ClusterShapeHitFilterESProducer = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  ComponentName = cms.string( "ClusterShapeHitFilter" ),
  appendToDataLabel = cms.string( "" )
)
EcalUnpackerWorkerESProducer = cms.ESProducer( "EcalUnpackerWorkerESProducer",
  ComponentName = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  DCCDataUnpacker = cms.PSet( 
    orderedDCCIdList = cms.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 ),
    tccUnpacking = cms.bool( False ),
    srpUnpacking = cms.bool( False ),
    syncCheck = cms.bool( False ),
    feIdCheck = cms.bool( True ),
    headerUnpacking = cms.bool( True ),
    orderedFedList = cms.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    feUnpacking = cms.bool( True ),
    forceKeepFRData = cms.bool( False ),
    memUnpacking = cms.bool( True )
  ),
  ElectronicsMapper = cms.PSet( 
    numbXtalTSamples = cms.uint32( 10 ),
    numbTriggerTSamples = cms.uint32( 1 )
  ),
  UncalibRHAlgo = cms.PSet(  Type = cms.string( "EcalUncalibRecHitWorkerWeights" ) ),
  CalibRHAlgo = cms.PSet( 
    flagsMapDBReco = cms.vint32( 0, 0, 0, 0, 4, -1, -1, -1, 4, 4, 7, 7, 7, 8, 9 ),
    Type = cms.string( "EcalRecHitWorkerSimple" ),
    killDeadChannels = cms.bool( True ),
    ChannelStatusToBeExcluded = cms.vint32( 10, 11, 12, 13, 14 ),
    laserCorrection = cms.bool( False ),
    EBLaserMIN = cms.double( 0.5 ),
    EELaserMIN = cms.double( 0.5 ),
    EBLaserMAX = cms.double( 2.0 ),
    EELaserMAX = cms.double( 3.0 )
  )
)
MaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "PropagatorWithMaterial" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Mass = cms.double( 0.105 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" )
)
MaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "PropagatorWithMaterialForHI" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Mass = cms.double( 0.139 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" )
)
OppositeMaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "PropagatorWithMaterialOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Mass = cms.double( 0.105 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" )
)
OppositeMaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Mass = cms.double( 0.139 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" )
)
SteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "anyDirection" ),
  useInTeslaFromMagField = cms.bool( False ),
  SetVBFPointer = cms.bool( False ),
  useMagVolumes = cms.bool( True ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  ApplyRadX0Correction = cms.bool( True ),
  AssumeNoMaterial = cms.bool( False ),
  NoErrorPropagation = cms.bool( False ),
  debug = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  sendLogWarning = cms.bool( False ),
  useTuningForL2Speed = cms.bool( False ),
  useEndcapShiftsInZ = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  appendToDataLabel = cms.string( "" )
)
TransientTrackBuilderESProducer = cms.ESProducer( "TransientTrackBuilderESProducer",
  ComponentName = cms.string( "TransientTrackBuilder" ),
  appendToDataLabel = cms.string( "" )
)
caloDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "CaloDetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
cosmicsNavigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "CosmicNavigationSchool" ),
  appendToDataLabel = cms.string( "" )
)
ecalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "EcalDetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.02 ),
  nEta = cms.int32( 300 ),
  nPhi = cms.int32( 360 ),
  includeBadChambers = cms.bool( False )
)
ecalSeverityLevel = cms.ESProducer( "EcalSeverityLevelESProducer",
  timeThresh = cms.double( 2.0 ),
  appendToDataLabel = cms.string( "" ),
  flagMask = cms.PSet( 
    kGood = cms.vstring( 'kGood' ),
    kProblematic = cms.vstring( 'kPoorReco',
      'kPoorCalib',
      'kNoisy',
      'kSaturated' ),
    kRecovered = cms.vstring( 'kLeadingEdgeRecovered',
      'kTowerRecovered' ),
    kTime = cms.vstring( 'kOutOfTime' ),
    kWeird = cms.vstring( 'kWeird',
      'kDiWeird' ),
    kBad = cms.vstring( 'kFaultyHardware',
      'kDead',
      'kKilled' )
  ),
  dbstatusMask = cms.PSet( 
    kGood = cms.vuint32( 0 ),
    kProblematic = cms.vuint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ),
    kRecovered = cms.vuint32(  ),
    kTime = cms.vuint32(  ),
    kWeird = cms.vuint32(  ),
    kBad = cms.vuint32( 11, 12, 13, 14, 15, 16 )
  )
)
hcalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HcalDetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
hcalRecAlgos = cms.ESProducer( "HcalRecAlgoESProducer",
  SeverityLevels = cms.VPSet( 
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring(  ),
      Level = cms.int32( 0 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring( 'HcalCellCaloTowerProb' ),
      Level = cms.int32( 1 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring( 'HSCP_R1R2',
  'HSCP_FracLeader',
  'HSCP_OuterEnergy',
  'HSCP_ExpFit',
  'ADCSaturationBit' ),
      ChannelStatus = cms.vstring(  ),
      Level = cms.int32( 5 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring( 'HBHEHpdHitMultiplicity',
  'HFDigiTime',
  'HBHEPulseShape',
  'HOBit',
  'HFInTimeWindow',
  'ZDCBit',
  'CalibrationBit',
  'TimingErrorBit' ),
      ChannelStatus = cms.vstring(  ),
      Level = cms.int32( 8 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring( 'HFLongShort',
  'HFS8S1Ratio',
  'HFPET' ),
      ChannelStatus = cms.vstring(  ),
      Level = cms.int32( 11 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring( 'HcalCellCaloTowerMask' ),
      Level = cms.int32( 12 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring( 'HcalCellHot' ),
      Level = cms.int32( 15 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring( 'HcalCellOff',
        'HcalCellDead' ),
      Level = cms.int32( 20 )
    )
  ),
  RecoveredRecHitBits = cms.vstring( 'TimingAddedBit',
    'TimingSubtractedBit' ),
  appendToDataLabel = cms.string( "" ),
  DropChannelStatusBits = cms.vstring( 'HcalCellMask',
    'HcalCellOff',
    'HcalCellDead' )
)
hltESPAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  ComponentName = cms.string( "hltESPAnalyticalPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  MaxDPhi = cms.double( 1.6 ),
  appendToDataLabel = cms.string( "" )
)
hltESPChi2EstimatorForRefit = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "hltESPChi2EstimatorForRefit" ),
  MaxChi2 = cms.double( 100000.0 ),
  nSigma = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
hltESPChi2MeasurementEstimator = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator" ),
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
hltESPChi2MeasurementEstimator16 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator16" ),
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
hltESPChi2MeasurementEstimator9 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator9" ),
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
hltESPCkf3HitTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltESPCkf3HitTrajectoryBuilder" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltESPCkf3HitTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
hltESPCkf3HitTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltESPCkf3HitTrajectoryFilter" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minPt = cms.double( 0.9 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  )
)
hltESPCkfTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltESPCkfTrajectoryBuilder" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltESPCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
hltESPCkfTrajectoryBuilderForHI = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltESPCkfTrajectoryBuilderForHI" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterialForHI" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTrackerForHI" ),
  trajectoryFilterName = cms.string( "hltESPCkfTrajectoryFilterForHI" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( False ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPCkfTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltESPCkfTrajectoryFilter" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minPt = cms.double( 0.9 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  )
)
hltESPCkfTrajectoryFilterForHI = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltESPCkfTrajectoryFilterForHI" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minimumNumberOfHits = cms.int32( 6 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 11.0 )
  )
)
hltESPDummyDetLayerGeometry = cms.ESProducer( "DetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPDummyDetLayerGeometry" ),
  appendToDataLabel = cms.string( "" )
)
hltESPESUnpackerWorker = cms.ESProducer( "ESUnpackerWorkerESProducer",
  ComponentName = cms.string( "hltESPESUnpackerWorker" ),
  appendToDataLabel = cms.string( "" ),
  DCCDataUnpacker = cms.PSet(  LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ) ),
  RHAlgo = cms.PSet( 
    ESRecoAlgo = cms.int32( 0 ),
    Type = cms.string( "ESRecHitWorker" )
  )
)
hltESPEcalRegionCablingESProducer = cms.ESProducer( "EcalRegionCablingESProducer",
  appendToDataLabel = cms.string( "" ),
  esMapping = cms.PSet(  LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ) )
)
hltESPFastSteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "anyDirection" ),
  useInTeslaFromMagField = cms.bool( False ),
  SetVBFPointer = cms.bool( False ),
  useMagVolumes = cms.bool( True ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  ApplyRadX0Correction = cms.bool( True ),
  AssumeNoMaterial = cms.bool( False ),
  NoErrorPropagation = cms.bool( False ),
  debug = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  sendLogWarning = cms.bool( False ),
  useTuningForL2Speed = cms.bool( True ),
  useEndcapShiftsInZ = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  appendToDataLabel = cms.string( "" )
)
hltESPFastSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useInTeslaFromMagField = cms.bool( False ),
  SetVBFPointer = cms.bool( False ),
  useMagVolumes = cms.bool( True ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  ApplyRadX0Correction = cms.bool( True ),
  AssumeNoMaterial = cms.bool( False ),
  NoErrorPropagation = cms.bool( False ),
  debug = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  sendLogWarning = cms.bool( False ),
  useTuningForL2Speed = cms.bool( True ),
  useEndcapShiftsInZ = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  appendToDataLabel = cms.string( "" )
)
hltESPFittingSmootherIT = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPFittingSmootherIT" ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  EstimateCut = cms.double( 10.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinNumberOfHits = cms.int32( 3 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
hltESPFittingSmootherRK = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPFittingSmootherRK" ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinNumberOfHits = cms.int32( 5 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPGlobalDetLayerGeometry = cms.ESProducer( "GlobalDetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPGlobalDetLayerGeometry" ),
  appendToDataLabel = cms.string( "" )
)
hltESPHIMixedLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltESPHIMixedLayerPairs" ),
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg',
    'FPix2_pos+TEC1_pos',
    'FPix2_pos+TEC2_pos',
    'TEC1_pos+TEC2_pos',
    'TEC2_pos+TEC3_pos',
    'FPix2_neg+TEC1_neg',
    'FPix2_neg+TEC2_neg',
    'TEC1_neg+TEC2_neg',
    'TEC2_neg+TEC3_neg' ),
  BPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0060 ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    useErrorsFromParam = cms.bool( True )
  ),
  FPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0036 ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    useErrorsFromParam = cms.bool( True )
  ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPHIPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltESPHIPixelLayerPairs" ),
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPHIPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltESPHIPixelLayerTriplets" ),
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPHITTRHBuilderWithoutRefit = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPHITTRHBuilderWithoutRefit" ),
  StripCPE = cms.string( "Fake" ),
  PixelCPE = cms.string( "Fake" ),
  Matcher = cms.string( "Fake" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPKFFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPKFFittingSmoother" ),
  Fitter = cms.string( "hltESPKFTrajectoryFitter" ),
  Smoother = cms.string( "hltESPKFTrajectorySmoother" ),
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinNumberOfHits = cms.int32( 5 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPKFFittingSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
  Fitter = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  Smoother = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinNumberOfHits = cms.int32( 5 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPKFFittingSmootherWithOutliersRejectionAndRK = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
  Fitter = cms.string( "hltESPRKFitter" ),
  Smoother = cms.string( "hltESPRKSmoother" ),
  EstimateCut = cms.double( 20.0 ),
  LogPixelProbabilityCut = cms.double( -14.0 ),
  MinNumberOfHits = cms.int32( 3 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
hltESPKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPKFTrajectoryFitter" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
hltESPKFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
hltESPKFTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPKFTrajectorySmoother" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
hltESPKFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
hltESPKFTrajectorySmootherForMuonTrackLoader = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
  Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
hltESPKFUpdator = cms.ESProducer( "KFUpdatorESProducer",
  ComponentName = cms.string( "hltESPKFUpdator" ),
  appendToDataLabel = cms.string( "" )
)
hltESPL3MuKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
  Propagator = cms.string( "hltESPSmartPropagatorAny" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
hltESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  ComponentName = cms.string( "hltESPMeasurementTracker" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  HitMatcher = cms.string( "StandardMatcher" ),
  Regional = cms.bool( True ),
  OnDemand = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  SiStripQualityLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  UseStripNoiseDB = cms.bool( False ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  stripClusterProducer = cms.string( "hltSiStripClusters" ),
  UseStripCablingDB = cms.bool( False ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  skipClusters = cms.InputTag( "" )
)
hltESPMeasurementTrackerForHI = cms.ESProducer( "MeasurementTrackerESProducer",
  ComponentName = cms.string( "hltESPMeasurementTrackerForHI" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  HitMatcher = cms.string( "StandardMatcher" ),
  Regional = cms.bool( False ),
  OnDemand = cms.bool( False ),
  UsePixelModuleQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  SiStripQualityLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  UseStripNoiseDB = cms.bool( False ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  stripClusterProducer = cms.string( "hltHISiStripClustersNonRegional" ),
  UseStripCablingDB = cms.bool( False ),
  pixelClusterProducer = cms.string( "hltHISiPixelClusters" ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripRawToDigi' ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 2 ),
      maxBad = cms.uint32( 4 )
    ),
    TID = cms.PSet( 
      maxBad = cms.uint32( 4 ),
      maxConsecutiveBad = cms.uint32( 2 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 2 ),
      maxBad = cms.uint32( 4 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 2 ),
      maxBad = cms.uint32( 4 )
    )
  ),
  skipClusters = cms.InputTag( "" )
)
hltESPMixedLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltESPMixedLayerPairs" ),
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg',
    'FPix2_pos+TEC1_pos',
    'FPix2_pos+TEC2_pos',
    'TEC1_pos+TEC2_pos',
    'TEC2_pos+TEC3_pos',
    'FPix2_neg+TEC1_neg',
    'FPix2_neg+TEC2_neg',
    'TEC1_neg+TEC2_neg',
    'TEC2_neg+TEC3_neg' ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPMuTrackJpsiTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltESPMuTrackJpsiTrajectoryBuilder" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltESPMuTrackJpsiTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPMuTrackJpsiTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltESPMuTrackJpsiTrajectoryFilter" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minPt = cms.double( 1.0 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 8 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  )
)
hltESPMuonCkfTrajectoryBuilder = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltESPMuonCkfTrajectoryBuilder" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltESPMuonCkfTrajectoryFilter" ),
  useSeedLayer = cms.bool( False ),
  rescaleErrorIfFail = cms.double( 1.0 ),
  deltaEta = cms.double( 0.1 ),
  deltaPhi = cms.double( 0.1 ),
  appendToDataLabel = cms.string( "" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( False ),
  alwaysUseInvalidHits = cms.bool( True )
)
hltESPMuonCkfTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltESPMuonCkfTrajectoryFilter" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minPt = cms.double( 0.9 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minimumNumberOfHits = cms.int32( 5 )
  )
)
hltESPMuonTransientTrackingRecHitBuilder = cms.ESProducer( "MuonTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
  appendToDataLabel = cms.string( "" )
)
hltESPPixelCPEGeneric = cms.ESProducer( "PixelCPEGenericESProducer",
  ComponentName = cms.string( "hltESPPixelCPEGeneric" ),
  appendToDataLabel = cms.string( "" ),
  TanLorentzAnglePerTesla = cms.double( 0.106 ),
  PixelErrorParametrization = cms.string( "NOTcmsim" ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  EdgeClusterErrorX = cms.double( 50.0 ),
  DoCosmics = cms.bool( False ),
  LoadTemplatesFromDB = cms.bool( True ),
  UseErrorsFromTemplates = cms.bool( True ),
  eff_charge_cut_highX = cms.double( 1.0 ),
  TruncatePixelCharge = cms.bool( True ),
  size_cutY = cms.double( 3.0 ),
  size_cutX = cms.double( 3.0 ),
  inflate_all_errors_no_trk_angle = cms.bool( False ),
  IrradiationBiasCorrection = cms.bool( False ),
  inflate_errors = cms.bool( False ),
  eff_charge_cut_lowX = cms.double( 0.0 ),
  eff_charge_cut_highY = cms.double( 1.0 ),
  EdgeClusterErrorY = cms.double( 85.0 ),
  eff_charge_cut_lowY = cms.double( 0.0 )
)
hltESPPixelCPETemplateReco = cms.ESProducer( "PixelCPETemplateRecoESProducer",
  ComponentName = cms.string( "hltESPPixelCPETemplateReco" ),
  appendToDataLabel = cms.string( "" ),
  TanLorentzAnglePerTesla = cms.double( 0.106 ),
  PixelErrorParametrization = cms.string( "" ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  DoCosmics = cms.bool( False ),
  UseClusterSplitter = cms.bool( False ),
  speed = cms.int32( -2 ),
  LoadTemplatesFromDB = cms.bool( True )
)
hltESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltESPPixelLayerPairs" ),
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltESPPixelLayerTriplets" ),
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPPixelLayerTripletsHITHB = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltESPPixelLayerTripletsHITHB" ),
  layerList = cms.vstring( 'BPix1+BPix2+BPix3' ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPPixelLayerTripletsHITHE = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltESPPixelLayerTripletsHITHE" ),
  layerList = cms.vstring( 'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltESPPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
  appendToDataLabel = cms.string( "" ),
  impactParameterType = cms.int32( 0 ),
  maximumDistanceToJetAxis = cms.double( 999999.0 ),
  deltaR = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  maxImpactParameterSig = cms.double( 999999.0 ),
  trackQualityClass = cms.string( "any" ),
  nthTrack = cms.int32( -1 ),
  maxImpactParameter = cms.double( 0.03 ),
  deltaRmin = cms.double( 0.0 )
)
hltESPRKTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPRKFitter" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
hltESPRKTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPRKSmoother" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
hltESPRungeKuttaTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Mass = cms.double( 0.105 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( True ),
  ptMin = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" )
)
hltESPSiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 )
)
hltESPSmartPropagator = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "hltESPSmartPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
  appendToDataLabel = cms.string( "" )
)
hltESPSmartPropagatorAny = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "hltESPSmartPropagatorAny" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  appendToDataLabel = cms.string( "" )
)
hltESPSmartPropagatorAnyOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  appendToDataLabel = cms.string( "" )
)
hltESPSmartPropagatorOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "hltESPSmartPropagatorOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
  appendToDataLabel = cms.string( "" )
)
hltESPSoftLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  appendToDataLabel = cms.string( "" ),
  distance = cms.double( 0.5 )
)
hltESPSoftLeptonByPt = cms.ESProducer( "LeptonTaggerByPtESProducer",
  appendToDataLabel = cms.string( "" ),
  ipSign = cms.string( "any" )
)
hltESPSteppingHelixPropagatorAlong = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useInTeslaFromMagField = cms.bool( False ),
  SetVBFPointer = cms.bool( False ),
  useMagVolumes = cms.bool( True ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  ApplyRadX0Correction = cms.bool( True ),
  AssumeNoMaterial = cms.bool( False ),
  NoErrorPropagation = cms.bool( False ),
  debug = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  sendLogWarning = cms.bool( False ),
  useTuningForL2Speed = cms.bool( False ),
  useEndcapShiftsInZ = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  appendToDataLabel = cms.string( "" )
)
hltESPSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useInTeslaFromMagField = cms.bool( False ),
  SetVBFPointer = cms.bool( False ),
  useMagVolumes = cms.bool( True ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  ApplyRadX0Correction = cms.bool( True ),
  AssumeNoMaterial = cms.bool( False ),
  NoErrorPropagation = cms.bool( False ),
  debug = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  sendLogWarning = cms.bool( False ),
  useTuningForL2Speed = cms.bool( False ),
  useEndcapShiftsInZ = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  appendToDataLabel = cms.string( "" )
)
hltESPStraightLinePropagator = cms.ESProducer( "StraightLinePropagatorESProducer",
  ComponentName = cms.string( "hltESPStraightLinePropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  appendToDataLabel = cms.string( "" )
)
hltESPTTRHBWithTrackAngle = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBWithTrackAngle" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPTTRHBuilderAngleAndTemplate = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  PixelCPE = cms.string( "hltESPPixelCPETemplateReco" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPTTRHBuilderPixelOnly = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBuilderPixelOnly" ),
  StripCPE = cms.string( "Fake" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPTTRHBuilderWithoutAngle4PixelTriplets = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" ),
  StripCPE = cms.string( "Fake" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPTrackCounting3D1st = cms.ESProducer( "TrackCountingESProducer",
  appendToDataLabel = cms.string( "" ),
  nthTrack = cms.int32( 1 ),
  impactParameterType = cms.int32( 0 ),
  deltaR = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 5.0 ),
  maximumDistanceToJetAxis = cms.double( 0.07 ),
  trackQualityClass = cms.string( "any" )
)
hltESPTrackCounting3D2nd = cms.ESProducer( "TrackCountingESProducer",
  appendToDataLabel = cms.string( "" ),
  nthTrack = cms.int32( 2 ),
  impactParameterType = cms.int32( 0 ),
  deltaR = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 5.0 ),
  maximumDistanceToJetAxis = cms.double( 0.07 ),
  trackQualityClass = cms.string( "any" )
)
hltESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltESPTrajectoryBuilderIT" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator9" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPTrajectoryBuilderL3 = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltESPTrajectoryBuilderL3" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterL3" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  appendToDataLabel = cms.string( "" ),
  fractionShared = cms.double( 0.5 ),
  allowSharedFirstHit = cms.bool( False )
)
hltESPTrajectoryCleanerBySharedSeeds = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedSeeds" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedSeeds" ),
  appendToDataLabel = cms.string( "" ),
  fractionShared = cms.double( 0.5 ),
  allowSharedFirstHit = cms.bool( True )
)
hltESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltESPTrajectoryFilterIT" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minPt = cms.double( 0.3 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 100 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  )
)
hltESPTrajectoryFilterL3 = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltESPTrajectoryFilterL3" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minPt = cms.double( 0.5 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 1000000000 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  )
)
hltESPTrajectoryFitterRK = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPTrajectoryFitterRK" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
hltESPTrajectorySmootherRK = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPTrajectorySmootherRK" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
hltESPbJetRegionalTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltESPbJetRegionalTrajectoryBuilder" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltESPbJetRegionalTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltESPbJetRegionalTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltESPbJetRegionalTrajectoryFilter" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minPt = cms.double( 1.0 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 8 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  )
)
hltHIAllESPCkf3HitTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltHIAllESPCkf3HitTrajectoryBuilder" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltESPCkf3HitTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
hltHIAllESPCkfTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltHIAllESPCkfTrajectoryBuilder" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltESPCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
hltHIAllESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  ComponentName = cms.string( "hltHIAllESPMeasurementTracker" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  HitMatcher = cms.string( "StandardMatcher" ),
  Regional = cms.bool( True ),
  OnDemand = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  SiStripQualityLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  UseStripNoiseDB = cms.bool( False ),
  stripLazyGetterProducer = cms.string( "hltHISiStripRawToClustersFacility" ),
  stripClusterProducer = cms.string( "hltHISiStripClusters" ),
  UseStripCablingDB = cms.bool( False ),
  pixelClusterProducer = cms.string( "hltHISiPixelClusters" ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  badStripCuts = cms.PSet( 
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  skipClusters = cms.InputTag( "" )
)
hltHIAllESPMuonCkfTrajectoryBuilder = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltHIAllESPMuonCkfTrajectoryBuilder" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltESPMuonCkfTrajectoryFilter" ),
  useSeedLayer = cms.bool( False ),
  rescaleErrorIfFail = cms.double( 1.0 ),
  deltaEta = cms.double( 0.1 ),
  deltaPhi = cms.double( 0.1 ),
  appendToDataLabel = cms.string( "" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( False ),
  alwaysUseInvalidHits = cms.bool( True )
)
hltHIAllESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltHIAllESPTrajectoryBuilderIT" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltIter1ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  ComponentName = cms.string( "hltIter1ESPMeasurementTracker" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  HitMatcher = cms.string( "StandardMatcher" ),
  Regional = cms.bool( True ),
  OnDemand = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  SiStripQualityLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  UseStripNoiseDB = cms.bool( False ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  stripClusterProducer = cms.string( "hltIter1SiStripClusters" ),
  UseStripCablingDB = cms.bool( False ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" )
)
hltIter1ESPPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltIter1ESPPixelLayerTriplets" ),
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TEC = cms.PSet(  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltIter1ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltIter1ESPTrajectoryBuilderIT" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltIter1ESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltIter1ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltIter1ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltIter1ESPTrajectoryFilterIT" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minPt = cms.double( 0.2 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 100 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  )
)
hltIter2ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  ComponentName = cms.string( "hltIter2ESPMeasurementTracker" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  HitMatcher = cms.string( "StandardMatcher" ),
  Regional = cms.bool( True ),
  OnDemand = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  SiStripQualityLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  UseStripNoiseDB = cms.bool( False ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  stripClusterProducer = cms.string( "hltIter2SiStripClusters" ),
  UseStripCablingDB = cms.bool( False ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  skipClusters = cms.InputTag( "hltIter2ClustersRefRemoval" )
)
hltIter2ESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltIter2ESPPixelLayerPairs" ),
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TEC = cms.PSet(  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
hltIter2ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltIter2ESPTrajectoryBuilderIT" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltIter2ESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltIter2ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltIter2ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltIter2ESPTrajectoryFilterIT" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minPt = cms.double( 0.3 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 100 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  )
)
hltIter3ESPLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltIter3ESPLayerTriplets" ),
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg',
    'BPix2+FPix1_pos+FPix2_pos',
    'BPix2+FPix1_neg+FPix2_neg',
    'FPix1_pos+FPix2_pos+TEC1_pos',
    'FPix1_neg+FPix2_neg+TEC1_neg',
    'FPix2_pos+TEC2_pos+TEC3_pos',
    'FPix2_neg+TEC2_neg+TEC3_neg',
    'BPix2+BPix3+TIB1',
    'BPix2+BPix3+TIB2',
    'BPix1+BPix3+TIB1',
    'BPix1+BPix3+TIB2',
    'BPix1+BPix2+TIB1',
    'BPix1+BPix2+TIB2' ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
hltIter3ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  ComponentName = cms.string( "hltIter3ESPMeasurementTracker" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  HitMatcher = cms.string( "StandardMatcher" ),
  Regional = cms.bool( True ),
  OnDemand = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  SiStripQualityLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  UseStripNoiseDB = cms.bool( False ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  stripClusterProducer = cms.string( "hltIter3SiStripClusters" ),
  UseStripCablingDB = cms.bool( False ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  skipClusters = cms.InputTag( "hltIter3ClustersRefRemoval" )
)
hltIter3ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltIter3ESPTrajectoryBuilderIT" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltIter3ESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltIter3ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltIter3ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltIter3ESPTrajectoryFilterIT" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minPt = cms.double( 0.3 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 0 ),
    maxNumberOfHits = cms.int32( 100 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  )
)
hltIter4ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  ComponentName = cms.string( "hltIter4ESPMeasurementTracker" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  HitMatcher = cms.string( "StandardMatcher" ),
  Regional = cms.bool( True ),
  OnDemand = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  SiStripQualityLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  UseStripNoiseDB = cms.bool( False ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  stripClusterProducer = cms.string( "hltIter4SiStripClusters" ),
  UseStripCablingDB = cms.bool( False ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  skipClusters = cms.InputTag( "hltIter4ClustersRefRemoval" )
)
hltIter4ESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltIter4ESPPixelLayerPairs" ),
  layerList = cms.vstring( 'TIB1+TIB2' ),
  BPix = cms.PSet(  ),
  FPix = cms.PSet(  ),
  TEC = cms.PSet(  ),
  TID = cms.PSet(  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
hltIter4ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltIter4ESPTrajectoryBuilderIT" ),
  updator = cms.string( "hltESPKFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltIter4ESPMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltIter4ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" ),
  minNrOfHitsForRebuild = cms.untracked.int32( 4 )
)
hltIter4ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltIter4ESPTrajectoryFilterIT" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minPt = cms.double( 0.3 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 0 ),
    maxNumberOfHits = cms.int32( 100 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 6 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  )
)
hoDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HODetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 30 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
muonDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "MuonDetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.125 ),
  nEta = cms.int32( 48 ),
  nPhi = cms.int32( 48 ),
  includeBadChambers = cms.bool( False )
)
navigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "SimpleNavigationSchool" ),
  appendToDataLabel = cms.string( "" )
)
preshowerDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "PreshowerDetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.1 ),
  nEta = cms.int32( 60 ),
  nPhi = cms.int32( 30 ),
  includeBadChambers = cms.bool( False )
)
siPixelQualityESProducer = cms.ESProducer( "SiPixelQualityESProducer",
  ListOfRecordToMerge = cms.VPSet( 
    cms.PSet(  record = cms.string( "SiPixelQualityFromDbRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiPixelDetVOffRcd" ),
      tag = cms.string( "" )
    )
  )
)
siPixelTemplateDBObjectESProducer = cms.ESProducer( "SiPixelTemplateDBObjectESProducer",
  appendToDataLabel = cms.string( "" )
)
siStripLorentzAngleDepESProducer = cms.ESProducer( "SiStripLorentzAngleDepESProducer",
  LatencyRecord = cms.PSet( 
    record = cms.string( "SiStripLatencyRcd" ),
    label = cms.untracked.string( "" )
  ),
  LorentzAngleDeconvMode = cms.PSet( 
    record = cms.string( "SiStripLorentzAngleRcd" ),
    label = cms.untracked.string( "deconvolution" )
  ),
  LorentzAnglePeakMode = cms.PSet( 
    record = cms.string( "SiStripLorentzAngleRcd" ),
    label = cms.untracked.string( "peak" )
  )
)

DTDataIntegrityTask = cms.Service( "DTDataIntegrityTask",
  getSCInfo = cms.untracked.bool( True ),
  fedIntegrityFolder = cms.untracked.string( "DT/FEDIntegrity_EvF" ),
  processingMode = cms.untracked.string( "HLT" )
)
UpdaterService = cms.Service( "UpdaterService",
)

hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
hltTriggerType = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 1 )
)
hltGtDigis = cms.EDProducer( "L1GlobalTriggerRawToDigi",
    DaqGtInputTag = cms.InputTag( "rawDataCollector" ),
    DaqGtFedId = cms.untracked.int32( 813 ),
    ActiveBoardsMask = cms.uint32( 0xffff ),
    UnpackBxInEvent = cms.int32( 5 ),
    Verbosity = cms.untracked.int32( 0 )
)
hltGctDigis = cms.EDProducer( "GctRawToDigi",
    inputLabel = cms.InputTag( "rawDataCollector" ),
    gctFedId = cms.untracked.int32( 745 ),
    hltMode = cms.bool( True ),
    numberOfGctSamplesToUnpack = cms.uint32( 1 ),
    numberOfRctSamplesToUnpack = cms.uint32( 1 ),
    unpackSharedRegions = cms.bool( False ),
    unpackerVersion = cms.uint32( 0 )
)
hltL1GtObjectMap = cms.EDProducer( "L1GlobalTrigger",
    GmtInputTag = cms.InputTag( "hltGtDigis" ),
    GctInputTag = cms.InputTag( "hltGctDigis" ),
    CastorInputTag = cms.InputTag( "castorL1Digis" ),
    ProduceL1GtDaqRecord = cms.bool( False ),
    ProduceL1GtEvmRecord = cms.bool( False ),
    ProduceL1GtObjectMapRecord = cms.bool( True ),
    WritePsbL1GtDaqRecord = cms.bool( False ),
    ReadTechnicalTriggerRecords = cms.bool( True ),
    EmulateBxInEvent = cms.int32( 1 ),
    AlternativeNrBxBoardDaq = cms.uint32( 0 ),
    AlternativeNrBxBoardEvm = cms.uint32( 0 ),
    BstLengthBytes = cms.int32( -1 ),
    AlgorithmTriggersUnprescaled = cms.bool( True ),
    AlgorithmTriggersUnmasked = cms.bool( False ),
    TechnicalTriggersUnprescaled = cms.bool( True ),
    TechnicalTriggersUnmasked = cms.bool( False ),
    TechnicalTriggersVetoUnmasked = cms.bool( True ),
    TechnicalTriggersInputTags = cms.VInputTag( 'simBscDigis' ),
    RecordLength = cms.vint32( 3, 0 )
)
hltL1extraParticles = cms.EDProducer( "L1ExtraParticlesProd",
    produceMuonParticles = cms.bool( True ),
    muonSource = cms.InputTag( "hltGtDigis" ),
    produceCaloParticles = cms.bool( True ),
    isolatedEmSource = cms.InputTag( 'hltGctDigis','isoEm' ),
    nonIsolatedEmSource = cms.InputTag( 'hltGctDigis','nonIsoEm' ),
    centralJetSource = cms.InputTag( 'hltGctDigis','cenJets' ),
    forwardJetSource = cms.InputTag( 'hltGctDigis','forJets' ),
    tauJetSource = cms.InputTag( 'hltGctDigis','tauJets' ),
    etTotalSource = cms.InputTag( "hltGctDigis" ),
    etHadSource = cms.InputTag( "hltGctDigis" ),
    etMissSource = cms.InputTag( "hltGctDigis" ),
    htMissSource = cms.InputTag( "hltGctDigis" ),
    hfRingEtSumsSource = cms.InputTag( "hltGctDigis" ),
    hfRingBitCountsSource = cms.InputTag( "hltGctDigis" ),
    centralBxOnly = cms.bool( True ),
    ignoreHtMiss = cms.bool( False )
)
hltScalersRawToDigi = cms.EDProducer( "ScalersRawToDigi",
    scalersInputTag = cms.InputTag( "rawDataCollector" )
)
hltOnlineBeamSpot = cms.EDProducer( "BeamSpotOnlineProducer",
    label = cms.InputTag( "hltScalersRawToDigi" ),
    src = cms.InputTag( "hltScalersRawToDigi" ),
    changeToCMSCoordinates = cms.bool( False ),
    maxRadius = cms.double( 2.0 ),
    maxZ = cms.double( 40.0 ),
    setSigmaZ = cms.double( 0.0 ),
    gtEvmLabel = cms.InputTag( "" )
)
hltOfflineBeamSpot = cms.EDProducer( "BeamSpotProducer" )
hltL1sL1ETM30BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM30_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIMET120 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltEcalRawToRecHitFacility = cms.EDProducer( "EcalRawToRecHitFacility",
    sourceTag = cms.InputTag( "rawDataCollector" ),
    workerName = cms.string( "" )
)
hltEcalRegionalRestFEDs = cms.EDProducer( "EcalRawToRecHitRoI",
    sourceTag = cms.InputTag( "hltEcalRawToRecHitFacility" ),
    type = cms.string( "all" ),
    doES = cms.bool( False ),
    sourceTag_es = cms.InputTag( "NotNeededoESfalse" ),
    MuJobPSet = cms.PSet(  ),
    JetJobPSet = cms.VPSet( 
    ),
    EmJobPSet = cms.VPSet( 
    ),
    CandJobPSet = cms.VPSet( 
    )
)
hltEcalRecHitAll = cms.EDProducer( "EcalRawToRecHitProducer",
    lazyGetterTag = cms.InputTag( "hltEcalRawToRecHitFacility" ),
    sourceTag = cms.InputTag( "hltEcalRegionalRestFEDs" ),
    splitOutput = cms.bool( True ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    rechitCollection = cms.string( "NotNeededsplitOutputTrue" ),
    cleaningConfig = cms.PSet( 
      tightenCrack_e1_double = cms.double( 2.0 ),
      tightenCrack_e6e2_double = cms.double( 3.0 ),
      e4e1Threshold_endcap = cms.double( 0.3 ),
      tightenCrack_e4e1_single = cms.double( 3.0 ),
      cThreshold_barrel = cms.double( 4.0 ),
      e4e1Threshold_barrel = cms.double( 0.08 ),
      tightenCrack_e1_single = cms.double( 2.0 ),
      e4e1_b_barrel = cms.double( -0.024 ),
      e4e1_a_barrel = cms.double( 0.04 ),
      ignoreOutOfTimeThresh = cms.double( 1000000.0 ),
      cThreshold_endcap = cms.double( 15.0 ),
      e4e1_b_endcap = cms.double( -0.0125 ),
      e4e1_a_endcap = cms.double( 0.02 ),
      e6e2thresh = cms.double( 0.04 ),
      cThreshold_double = cms.double( 10.0 ),
      swissCrossThreshold = cms.double( 0.95 ),
      recHitThreshold = cms.double( 4.0 ),
      useieta85 = cms.bool( True )
    )
)
hltHcalDigis = cms.EDProducer( "HcalRawToDigi",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    UnpackCalib = cms.untracked.bool( True ),
    UnpackZDC = cms.untracked.bool( True ),
    firstSample = cms.int32( 0 ),
    lastSample = cms.int32( 9 ),
    FilterDataQuality = cms.bool( True )
)
hltHbhereco = cms.EDProducer( "HcalHitReconstructor",
    correctForTimeslew = cms.bool( True ),
    correctForPhaseContainment = cms.bool( True ),
    correctionPhaseNS = cms.double( 13.0 ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    correctTiming = cms.bool( False ),
    setNoiseFlags = cms.bool( False ),
    setHSCPFlags = cms.bool( False ),
    setSaturationFlags = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    dropZSmarkedPassed = cms.bool( True ),
    firstAuxTS = cms.int32( 4 ),
    firstSample = cms.int32( 4 ),
    samplesToAdd = cms.int32( 4 ),
    tsFromDB = cms.bool( True ),
    useLeakCorrection = cms.bool( False ),
    Subdetector = cms.string( "HBHE" ),
    setTimingShapedCutsFlags = cms.bool( False ),
    digiTimeFromDB = cms.bool( False ),
    hfTimingTrustParameters = cms.PSet( 
      hfTimingTrustLevel2 = cms.int32( 4 ),
      hfTimingTrustLevel1 = cms.int32( 1 )
    ),
    digistat = cms.PSet(  ),
    HFInWindowStat = cms.PSet(  ),
    S9S1stat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      flagsToSkip = cms.int32( 24 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      long_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      isS8S1 = cms.bool( False ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    S8S1stat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      shortEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      flagsToSkip = cms.int32( 16 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      longEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      long_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      isS8S1 = cms.bool( True ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    PETstat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_R_29 = cms.vdouble( 0.8 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      flagsToSkip = cms.int32( 0 ),
      short_R = cms.vdouble( 0.8 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_R_29 = cms.vdouble( 0.8 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      long_R = cms.vdouble( 0.98 ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    timingshapedcutsParameters = cms.PSet( 
      ignorelowest = cms.bool( True ),
      win_offset = cms.double( 0.0 ),
      ignorehighest = cms.bool( False ),
      win_gain = cms.double( 1.0 ),
      tfilterEnvelope = cms.vdouble( 4.0, 12.04, 13.0, 10.56, 23.5, 8.82, 37.0, 7.38, 56.0, 6.3, 81.0, 5.64, 114.5, 5.44, 175.5, 5.38, 350.5, 5.14 )
    ),
    flagParameters = cms.PSet( 
      nominalPedestal = cms.double( 3.0 ),
      hitMultiplicityThreshold = cms.int32( 17 ),
      hitEnergyMinimum = cms.double( 1.0 ),
      pulseShapeParameterSets = cms.VPSet( 
        cms.PSet(  pulseShapeParameters = cms.vdouble( 0.0, 100.0, -50.0, 0.0, -15.0, 0.15 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 100.0, 2000.0, -50.0, 0.0, -5.0, 0.05 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 2000.0, 1000000.0, -50.0, 0.0, 95.0, 0.0 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( -1000000.0, 1000000.0, 45.0, 0.1, 1000000.0, 0.0 )        )
      )
    ),
    hscpParameters = cms.PSet( 
      slopeMax = cms.double( -0.6 ),
      r1Max = cms.double( 1.0 ),
      r1Min = cms.double( 0.15 ),
      TimingEnergyThreshold = cms.double( 30.0 ),
      slopeMin = cms.double( -1.5 ),
      outerMin = cms.double( 0.0 ),
      outerMax = cms.double( 0.1 ),
      fracLeaderMin = cms.double( 0.4 ),
      r2Min = cms.double( 0.1 ),
      r2Max = cms.double( 0.5 ),
      fracLeaderMax = cms.double( 0.7 )
    ),
    pulseShapeParameters = cms.PSet(  ),
    firstAuxOffset = cms.int32( 0 )
)
hltHfreco = cms.EDProducer( "HcalHitReconstructor",
    correctForTimeslew = cms.bool( False ),
    correctForPhaseContainment = cms.bool( False ),
    correctionPhaseNS = cms.double( 13.0 ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    correctTiming = cms.bool( False ),
    setNoiseFlags = cms.bool( False ),
    setHSCPFlags = cms.bool( False ),
    setSaturationFlags = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    dropZSmarkedPassed = cms.bool( True ),
    firstAuxTS = cms.int32( 1 ),
    firstSample = cms.int32( 2 ),
    samplesToAdd = cms.int32( 2 ),
    tsFromDB = cms.bool( True ),
    useLeakCorrection = cms.bool( False ),
    Subdetector = cms.string( "HF" ),
    setTimingShapedCutsFlags = cms.bool( False ),
    digiTimeFromDB = cms.bool( False ),
    hfTimingTrustParameters = cms.PSet( 
      hfTimingTrustLevel2 = cms.int32( 4 ),
      hfTimingTrustLevel1 = cms.int32( 1 )
    ),
    digistat = cms.PSet( 
      HFdigiflagFirstSample = cms.int32( 1 ),
      HFdigiflagMinEthreshold = cms.double( 40.0 ),
      HFdigiflagSamplesToAdd = cms.int32( 3 ),
      HFdigiflagExpectedPeak = cms.int32( 2 ),
      HFdigiflagCoef = cms.vdouble( 0.93, -0.012667, -0.38275 )
    ),
    HFInWindowStat = cms.PSet( 
      hflongEthresh = cms.double( 40.0 ),
      hflongMinWindowTime = cms.vdouble( -10.0 ),
      hfshortEthresh = cms.double( 40.0 ),
      hflongMaxWindowTime = cms.vdouble( 10.0 ),
      hfshortMaxWindowTime = cms.vdouble( 10.0 ),
      hfshortMinWindowTime = cms.vdouble( -12.0 )
    ),
    S9S1stat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      flagsToSkip = cms.int32( 24 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      long_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      isS8S1 = cms.bool( False ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    S8S1stat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      shortEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      flagsToSkip = cms.int32( 16 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      longEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      long_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      isS8S1 = cms.bool( True ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    PETstat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_R_29 = cms.vdouble( 0.8 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      flagsToSkip = cms.int32( 0 ),
      short_R = cms.vdouble( 0.8 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_R_29 = cms.vdouble( 0.8 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      long_R = cms.vdouble( 0.98 ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    timingshapedcutsParameters = cms.PSet( 
      ignorelowest = cms.bool( True ),
      win_offset = cms.double( 0.0 ),
      ignorehighest = cms.bool( False ),
      win_gain = cms.double( 1.0 ),
      tfilterEnvelope = cms.vdouble( 4.0, 12.04, 13.0, 10.56, 23.5, 8.82, 37.0, 7.38, 56.0, 6.3, 81.0, 5.64, 114.5, 5.44, 175.5, 5.38, 350.5, 5.14 )
    ),
    flagParameters = cms.PSet( 
      nominalPedestal = cms.double( 3.0 ),
      hitMultiplicityThreshold = cms.int32( 17 ),
      hitEnergyMinimum = cms.double( 1.0 ),
      pulseShapeParameterSets = cms.VPSet( 
        cms.PSet(  pulseShapeParameters = cms.vdouble( 0.0, 100.0, -50.0, 0.0, -15.0, 0.15 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 100.0, 2000.0, -50.0, 0.0, -5.0, 0.05 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 2000.0, 1000000.0, -50.0, 0.0, 95.0, 0.0 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( -1000000.0, 1000000.0, 45.0, 0.1, 1000000.0, 0.0 )        )
      )
    ),
    hscpParameters = cms.PSet( 
      slopeMax = cms.double( -0.6 ),
      r1Max = cms.double( 1.0 ),
      r1Min = cms.double( 0.15 ),
      TimingEnergyThreshold = cms.double( 30.0 ),
      slopeMin = cms.double( -1.5 ),
      outerMin = cms.double( 0.0 ),
      outerMax = cms.double( 0.1 ),
      fracLeaderMin = cms.double( 0.4 ),
      r2Min = cms.double( 0.1 ),
      r2Max = cms.double( 0.5 ),
      fracLeaderMax = cms.double( 0.7 )
    ),
    pulseShapeParameters = cms.PSet(  ),
    firstAuxOffset = cms.int32( 0 )
)
hltHoreco = cms.EDProducer( "HcalHitReconstructor",
    correctForTimeslew = cms.bool( True ),
    correctForPhaseContainment = cms.bool( True ),
    correctionPhaseNS = cms.double( 13.0 ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    correctTiming = cms.bool( False ),
    setNoiseFlags = cms.bool( False ),
    setHSCPFlags = cms.bool( False ),
    setSaturationFlags = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    dropZSmarkedPassed = cms.bool( True ),
    firstAuxTS = cms.int32( 4 ),
    firstSample = cms.int32( 4 ),
    samplesToAdd = cms.int32( 4 ),
    tsFromDB = cms.bool( True ),
    useLeakCorrection = cms.bool( False ),
    Subdetector = cms.string( "HO" ),
    setTimingShapedCutsFlags = cms.bool( False ),
    digiTimeFromDB = cms.bool( False ),
    hfTimingTrustParameters = cms.PSet( 
      hfTimingTrustLevel2 = cms.int32( 4 ),
      hfTimingTrustLevel1 = cms.int32( 1 )
    ),
    digistat = cms.PSet(  ),
    HFInWindowStat = cms.PSet(  ),
    S9S1stat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      HcalAcceptSeverityLevel = cms.int32( 9 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      flagsToSkip = cms.int32( 24 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      long_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      isS8S1 = cms.bool( False )
    ),
    S8S1stat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      HcalAcceptSeverityLevel = cms.int32( 9 ),
      shortEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      flagsToSkip = cms.int32( 16 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      longEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      long_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      isS8S1 = cms.bool( True )
    ),
    PETstat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_R_29 = cms.vdouble( 0.8 ),
      HcalAcceptSeverityLevel = cms.int32( 9 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      flagsToSkip = cms.int32( 0 ),
      long_R_29 = cms.vdouble( 0.8 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_R = cms.vdouble( 0.8 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      long_R = cms.vdouble( 0.98 )
    ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    timingshapedcutsParameters = cms.PSet( 
      ignorelowest = cms.bool( True ),
      win_offset = cms.double( 0.0 ),
      ignorehighest = cms.bool( False ),
      win_gain = cms.double( 1.0 ),
      tfilterEnvelope = cms.vdouble( 4.0, 12.04, 13.0, 10.56, 23.5, 8.82, 37.0, 7.38, 56.0, 6.3, 81.0, 5.64, 114.5, 5.44, 175.5, 5.38, 350.5, 5.14 )
    ),
    flagParameters = cms.PSet( 
      nominalPedestal = cms.double( 3.0 ),
      hitMultiplicityThreshold = cms.int32( 17 ),
      hitEnergyMinimum = cms.double( 1.0 ),
      pulseShapeParameterSets = cms.VPSet( 
        cms.PSet(  pulseShapeParameters = cms.vdouble( 0.0, 100.0, -50.0, 0.0, -15.0, 0.15 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 100.0, 2000.0, -50.0, 0.0, -5.0, 0.05 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 2000.0, 1000000.0, -50.0, 0.0, 95.0, 0.0 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( -1000000.0, 1000000.0, 45.0, 0.1, 1000000.0, 0.0 )        )
      )
    ),
    hscpParameters = cms.PSet( 
      slopeMax = cms.double( -0.6 ),
      r1Max = cms.double( 1.0 ),
      r1Min = cms.double( 0.15 ),
      TimingEnergyThreshold = cms.double( 30.0 ),
      slopeMin = cms.double( -1.5 ),
      outerMin = cms.double( 0.0 ),
      outerMax = cms.double( 0.1 ),
      fracLeaderMin = cms.double( 0.4 ),
      r2Min = cms.double( 0.1 ),
      r2Max = cms.double( 0.5 ),
      fracLeaderMax = cms.double( 0.7 )
    ),
    pulseShapeParameters = cms.PSet(  ),
    firstAuxOffset = cms.int32( 0 )
)
hltTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.07 ),
    EEThreshold = cms.double( 0.3 ),
    UseEtEBTreshold = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    UseSymEBTreshold = cms.bool( False ),
    UseSymEETreshold = cms.bool( False ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.7 ),
    HESThreshold = cms.double( 0.8 ),
    HEDThreshold = cms.double( 0.8 ),
    HOThreshold0 = cms.double( 3.5 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HF1Threshold = cms.double( 0.5 ),
    HF2Threshold = cms.double( 0.85 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0E-99 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHBDepth = cms.double( 0.2 ),
    MomHEDepth = cms.double( 0.4 ),
    MomEBDepth = cms.double( 0.3 ),
    MomEEDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    AllowMissingInputs = cms.bool( False ),
    HcalAcceptSeverityLevel = cms.uint32( 9 ),
    UseHcalRecoveredHits = cms.bool( False ),
    UseEcalRecoveredHits = cms.bool( False ),
    UseRejectedHitsOnly = cms.bool( False ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    UseRejectedRecoveredHcalHits = cms.bool( False ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    EBGrid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    EEWeights = cms.vdouble(  ),
    HBGrid = cms.vdouble(  ),
    HBWeights = cms.vdouble(  ),
    HESGrid = cms.vdouble(  ),
    HESWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HOGrid = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    HF1Grid = cms.vdouble(  ),
    HF1Weights = cms.vdouble(  ),
    HF2Grid = cms.vdouble(  ),
    HF2Weights = cms.vdouble(  ),
    ecalInputs = cms.VInputTag( 'hltEcalRecHitAll:EcalRecHitsEB','hltEcalRecHitAll:EcalRecHitsEE' ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kTime',
      'kWeird',
      'kBad' ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  )
)
hltMet = cms.EDProducer( "METProducer",
    src = cms.InputTag( "hltTowerMakerForAll" ),
    InputType = cms.string( "CandidateCollection" ),
    METType = cms.string( "CaloMET" ),
    alias = cms.string( "RawCaloMET" ),
    globalThreshold = cms.double( 0.3 ),
    noHF = cms.bool( False ),
    calculateSignificance = cms.bool( False ),
    onlyFiducialParticles = cms.bool( False ),
    jets = cms.InputTag( "unused" ),
    rf_type = cms.int32( 0 ),
    correctShowerTracks = cms.bool( False ),
    HO_EtResPar = cms.vdouble( 0.0, 1.3, 0.0050 ),
    HF_EtResPar = cms.vdouble( 0.0, 1.82, 0.09 ),
    HB_PhiResPar = cms.vdouble( 0.02511 ),
    HE_PhiResPar = cms.vdouble( 0.02511 ),
    EE_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    EB_PhiResPar = cms.vdouble( 0.00502 ),
    EE_PhiResPar = cms.vdouble( 0.02511 ),
    HB_EtResPar = cms.vdouble( 0.0, 1.22, 0.05 ),
    EB_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    HF_PhiResPar = cms.vdouble( 0.05022 ),
    HE_EtResPar = cms.vdouble( 0.0, 1.3, 0.05 ),
    HO_PhiResPar = cms.vdouble( 0.02511 )
)
hltHIMET120 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTags = cms.bool( True ),
    MinPt = cms.double( 120.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltSiStripRawToDigi = cms.EDProducer( "SiStripRawToDigiModule",
    ProductLabel = cms.InputTag( "rawDataCollector" ),
    AppendedBytes = cms.int32( 0 ),
    UseDaqRegister = cms.bool( False ),
    UseFedKey = cms.bool( False ),
    UnpackBadChannels = cms.bool( False ),
    MarkModulesOnMissingFeds = cms.bool( True ),
    TriggerFedId = cms.int32( 0 ),
    UnpackCommonModeValues = cms.bool( False ),
    DoAllCorruptBufferChecks = cms.bool( False ),
    ErrorThreshold = cms.uint32( 7174 )
)
hltSiStripZeroSuppression = cms.EDProducer( "SiStripZeroSuppression",
    Algorithms = cms.PSet( 
      CutToAvoidSignal = cms.double( 2.0 ),
      PedestalSubtractionFedMode = cms.bool( False ),
      Fraction = cms.double( 0.2 ),
      minStripsToFit = cms.uint32( 4 ),
      consecThreshold = cms.uint32( 5 ),
      hitStripThreshold = cms.uint32( 40 ),
      Deviation = cms.uint32( 25 ),
      CommonModeNoiseSubtractionMode = cms.string( "IteratedMedian" ),
      TruncateInSuppressor = cms.bool( True ),
      restoreThreshold = cms.double( 0.5 ),
      APVInspectMode = cms.string( "BaselineFollower" ),
      ForceNoRestore = cms.bool( False ),
      useRealMeanCM = cms.bool( False ),
      DeltaCMThreshold = cms.uint32( 20 ),
      nSigmaNoiseDerTh = cms.uint32( 4 ),
      nSaturatedStrip = cms.uint32( 2 ),
      SiStripFedZeroSuppressionMode = cms.uint32( 4 ),
      APVRestoreMode = cms.string( "BaselineFollower" ),
      distortionThreshold = cms.uint32( 20 ),
      Iterations = cms.int32( 3 ),
      nSmooth = cms.uint32( 9 ),
      SelfSelectRestoreAlgo = cms.bool( False ),
      doAPVRestore = cms.bool( True ),
      useCMMeanMap = cms.bool( False ),
      ApplyBaselineCleaner = cms.bool( True ),
      MeanCM = cms.int32( 0 ),
      CleaningSequence = cms.uint32( 1 ),
      slopeX = cms.int32( 3 ),
      slopeY = cms.int32( 4 ),
      ApplyBaselineRejection = cms.bool( True ),
      filteredBaselineMax = cms.double( 6.0 ),
      filteredBaselineDerivativeSumSquare = cms.double( 30.0 )
    ),
    doAPVRestore = cms.bool( True ),
    produceCalculatedBaseline = cms.bool( False ),
    mergeCollections = cms.bool( False ),
    RawDigiProducersList = cms.VInputTag( 'hltSiStripRawToDigi:VirginRaw','hltSiStripRawToDigi:ProcessedRaw','hltSiStripRawToDigi:ScopeMode' ),
    storeCM = cms.bool( False ),
    useCMMeanMap = cms.bool( False ),
    produceRawDigis = cms.bool( True ),
    NumberOfModuleSkippedBeforeForcingRestore = cms.uint32( 20 ),
    storeInZScollBadAPV = cms.bool( True ),
    fixCM = cms.bool( False ),
    produceBaselinePoints = cms.bool( False )
)
hltSiStripDigiToZSRaw = cms.EDProducer( "SiStripDigiToRawModule",
    InputModuleLabel = cms.string( "hltSiStripZeroSuppression" ),
    InputDigiLabel = cms.string( "VirginRaw" ),
    FedReadoutMode = cms.string( "ZERO_SUPPRESSED" ),
    UseFedKey = cms.bool( False ),
    UseWrongDigiType = cms.bool( False )
)
hltSiStripRawDigiToVirginRaw = cms.EDProducer( "SiStripDigiToRawModule",
    InputModuleLabel = cms.string( "hltSiStripZeroSuppression" ),
    InputDigiLabel = cms.string( "VirginRaw" ),
    FedReadoutMode = cms.string( "VIRGIN_RAW" ),
    UseFedKey = cms.bool( False ),
    UseWrongDigiType = cms.bool( False )
)
virginRawDataRepacker = cms.EDProducer( "RawDataCollectorByLabel",
    RawCollectionList = cms.VInputTag( 'hltSiStripRawDigiToVirginRaw' )
)
rawDataRepacker = cms.EDProducer( "RawDataCollectorByLabel",
    RawCollectionList = cms.VInputTag( 'hltSiStripDigiToZSRaw','source','rawDataCollector' )
)
hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
hltL1sL1ETM50BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM50_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIMET200 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIMET200 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTags = cms.bool( True ),
    MinPt = cms.double( 200.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPreHIMET220 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIMET220 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTags = cms.bool( True ),
    MinPt = cms.double( 220.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPreHIPhysics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltCalibrationEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 2 )
)
hltPreHIDTCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltDTCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780 )
)
hltPreHIEcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltEcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 )
)
hltPreHIHcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHcalCalibTypeFilter = cms.EDFilter( "HLTHcalCalibTypeFilter",
    InputTag = cms.InputTag( "rawDataCollector" ),
    CalibTypes = cms.vint32( 1, 2, 3, 4, 5, 6 )
)
hltHcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731 )
)
hltL1sHIZeroBias = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sL1BptxXOR = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxXOR" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIZeroBiasXOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIZeroBiasXOR = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxPlusANDMinus OR L1_BptxXOR" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIZeroBiasPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltSiPixelDigis = cms.EDProducer( "SiPixelRawToDigi",
    IncludeErrors = cms.bool( False ),
    UseQualityInfo = cms.bool( False ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    ErrorList = cms.vint32(  )
)
hltHISiPixelClusters = cms.EDProducer( "SiPixelClusterProducer",
    src = cms.InputTag( "hltSiPixelDigis" ),
    maxNumberOfClusters = cms.int32( -1 ),
    payloadType = cms.string( "HLT" ),
    ChannelThreshold = cms.int32( 1000 ),
    SeedThreshold = cms.int32( 1000 ),
    ClusterThreshold = cms.double( 4000.0 ),
    VCaltoElectronGain = cms.int32( 65 ),
    VCaltoElectronOffset = cms.int32( -414 ),
    MissCalibrate = cms.untracked.bool( True ),
    SplitClusters = cms.bool( False )
)
hltHISiPixelRecHits = cms.EDProducer( "SiPixelRecHitConverter",
    src = cms.InputTag( "hltHISiPixelClusters" ),
    CPE = cms.string( "hltESPPixelCPEGeneric" )
)
hltHIPixelClusterVertices = cms.EDProducer( "HIPixelClusterVtxProducer",
    pixelRecHits = cms.string( "hltHISiPixelRecHits" ),
    minZ = cms.double( -20.0 ),
    maxZ = cms.double( 20.05 ),
    zStep = cms.double( 0.1 )
)
hltPixelTracksForHITrackTrigger = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( False ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "HITrackingRegionForPrimaryVtxProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        doVariablePtMin = cms.bool( True ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFixedError = cms.bool( True ),
        directionYCoord = cms.double( 1.0 ),
        sigmaZVertex = cms.double( 3.0 ),
        fixedError = cms.double( 2.0 ),
        directionXCoord = cms.double( 1.0 ),
        directionZCoord = cms.double( 0.0 ),
        VertexCollection = cms.InputTag( "hltHIPixelClusterVertices" ),
        ptMin = cms.double( 0.7 ),
        siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" ),
        nSigmaZ = cms.double( 3.0 ),
        useFoundVertices = cms.bool( True ),
        originRadius = cms.double( 0.1 )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        maxElement = cms.uint32( 100000 ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.06 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.06 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      ),
      SeedingLayers = cms.string( "hltESPHIPixelLayerTriplets" )
    ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" )
    ),
    FilterPSet = cms.PSet( 
      doVariablePtMin = cms.bool( True ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIProtoTrackFilter" ),
      ptMin = cms.double( 1.0 ),
      siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" ),
      tipMax = cms.double( 1.0 )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) )
)
hltPixelCandsForHITrackTrigger = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPixelTracksForHITrackTrigger" ),
    particleType = cms.string( "pi+" )
)
hltHISinglePixelTrackFilter = cms.EDFilter( "HLTPixlMBFilt",
    pixlTag = cms.InputTag( "hltPixelCandsForHITrackTrigger" ),
    MinPt = cms.double( 0.0 ),
    MinTrks = cms.uint32( 1 ),
    MinSep = cms.double( 1.0 )
)
hltL1sHIMinBiasBSC = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BscMinBiasThreshold1_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIMinBiasBSC = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIMinBiasBSCOR = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BscMinBiasOR_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIMinBiasBSCOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIMinBiasHF = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HcalHfCoincidencePm_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIMinBiasHF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIMinBiasHfOr = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HcalHfMmOrPpOrPm_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIMinBiasHfOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIMinBiasHfOrBSC = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HcalHfCoincPmORBscMinBiasThresh1_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIMinBiasHfOrBSC = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltPreHIMinBiasPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIMinBiasZDC = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZdcCaloPlus_ZdcCaloMinus_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIMinBiasZDCCalo = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIMinBiasZDCCaloPlusOrMinus = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZdcCaloPlus_BptxAND OR L1_ZdcCaloMinus_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIMinBiasZDCCaloPlusOrMinus = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIMinBiasZDCPixelSingleTrack = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZdcCaloPlus_ZdcCaloMinus_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIMinBiasZDCPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIMinBiasZDCPlusOrMinusPixelSingleTrack = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZdcCaloPlus_BptxAND OR L1_ZdcCaloMinus_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIMinBiasZDCPlusOrMinusPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltPreHIBptxXOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sL1BptxXORBscMinBiasOR = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxXOR_BscMinBiasOR" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIL1AlgoBptxXORBSCOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sL1DoubleMuOpenBptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "( L1_DoubleMuOpen_BptxAND OR L1_ETT2000 ) AND ( L1_HcalHfCoincPmORBscMinBiasThresh1_BptxAND_instance1 OR L1_NotBsc2_BscMinBiasOR OR L1_NotBsc2_HcalHfMmOrPpOrPm )" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIL1DoubleMuOpen = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDoubleMuLevel1PathL1OpenFiltered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpenBptxAND" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    ExcludeSingleSegmentCSC = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    saveTags = cms.bool( True ),
    SelectQualities = cms.vint32(  )
)
hltPreHIL1DoubleMuOpenHighQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDoubleMuLevel1PathL1HighQFiltered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpenBptxAND" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    ExcludeSingleSegmentCSC = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    saveTags = cms.bool( True ),
    SelectQualities = cms.vint32( 7, 6, 5, 0, 0, 0, 0, 0 )
)
hltL1sL1SingleMu3BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "( L1_SingleMu3_BptxAND OR L1_ETT2000 ) AND ( L1_HcalHfCoincPmORBscMinBiasThresh1_BptxAND_instance1 OR L1_NotBsc2_BscMinBiasOR OR L1_NotBsc2_HcalHfMmOrPpOrPm )" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIL2Mu3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIL1SingleMu3withBptxANDL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3BptxAND" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    ExcludeSingleSegmentCSC = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    saveTags = cms.bool( False ),
    SelectQualities = cms.vint32(  )
)
hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
    dataType = cms.string( "DDU" ),
    fedbyType = cms.bool( False ),
    inputLabel = cms.InputTag( "rawDataCollector" ),
    useStandardFEDid = cms.bool( True ),
    dqmOnly = cms.bool( False ),
    rosParameters = cms.PSet(  ),
    readOutParameters = cms.PSet( 
      debug = cms.untracked.bool( False ),
      rosParameters = cms.PSet( 
        writeSC = cms.untracked.bool( True ),
        readingDDU = cms.untracked.bool( True ),
        performDataIntegrityMonitor = cms.untracked.bool( False ),
        readDDUIDfromDDU = cms.untracked.bool( True ),
        debug = cms.untracked.bool( False ),
        localDAQ = cms.untracked.bool( False )
      ),
      localDAQ = cms.untracked.bool( False ),
      performDataIntegrityMonitor = cms.untracked.bool( False )
    )
)
hltDt1DRecHits = cms.EDProducer( "DTRecHitProducer",
    dtDigiLabel = cms.InputTag( "hltMuonDTDigis" ),
    recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
    recAlgoConfig = cms.PSet( 
      tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
      minTime = cms.double( -3.0 ),
      stepTwoFromDigi = cms.bool( False ),
      doVdriftCorr = cms.bool( False ),
      debug = cms.untracked.bool( False ),
      maxTime = cms.double( 420.0 ),
      tTrigModeConfig = cms.PSet( 
        vPropWire = cms.double( 24.4 ),
        doTOFCorrection = cms.bool( True ),
        tofCorrType = cms.int32( 0 ),
        wirePropCorrType = cms.int32( 0 ),
        tTrigLabel = cms.string( "" ),
        doWirePropCorrection = cms.bool( True ),
        doT0Correction = cms.bool( True ),
        debug = cms.untracked.bool( False )
      )
    )
)
hltDt4DSegments = cms.EDProducer( "DTRecSegment4DProducer",
    debug = cms.untracked.bool( False ),
    recHits1DLabel = cms.InputTag( "hltDt1DRecHits" ),
    recHits2DLabel = cms.InputTag( "dt2DSegments" ),
    Reco4DAlgoName = cms.string( "DTCombinatorialPatternReco4D" ),
    Reco4DAlgoConfig = cms.PSet( 
      segmCleanerMode = cms.int32( 2 ),
      Reco2DAlgoName = cms.string( "DTCombinatorialPatternReco" ),
      recAlgoConfig = cms.PSet( 
        tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
        minTime = cms.double( -3.0 ),
        stepTwoFromDigi = cms.bool( False ),
        doVdriftCorr = cms.bool( False ),
        debug = cms.untracked.bool( False ),
        maxTime = cms.double( 420.0 ),
        tTrigModeConfig = cms.PSet( 
          vPropWire = cms.double( 24.4 ),
          doTOFCorrection = cms.bool( True ),
          tofCorrType = cms.int32( 0 ),
          wirePropCorrType = cms.int32( 0 ),
          tTrigLabel = cms.string( "" ),
          doWirePropCorrection = cms.bool( True ),
          doT0Correction = cms.bool( True ),
          debug = cms.untracked.bool( False )
        )
      ),
      nSharedHitsMax = cms.int32( 2 ),
      hit_afterT0_resolution = cms.double( 0.03 ),
      Reco2DAlgoConfig = cms.PSet( 
        segmCleanerMode = cms.int32( 2 ),
        recAlgoConfig = cms.PSet( 
          tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
          minTime = cms.double( -3.0 ),
          stepTwoFromDigi = cms.bool( False ),
          doVdriftCorr = cms.bool( False ),
          debug = cms.untracked.bool( False ),
          maxTime = cms.double( 420.0 ),
          tTrigModeConfig = cms.PSet( 
            vPropWire = cms.double( 24.4 ),
            doTOFCorrection = cms.bool( True ),
            tofCorrType = cms.int32( 0 ),
            wirePropCorrType = cms.int32( 0 ),
            tTrigLabel = cms.string( "" ),
            doWirePropCorrection = cms.bool( True ),
            doT0Correction = cms.bool( True ),
            debug = cms.untracked.bool( False )
          )
        ),
        nSharedHitsMax = cms.int32( 2 ),
        AlphaMaxPhi = cms.double( 1.0 ),
        hit_afterT0_resolution = cms.double( 0.03 ),
        MaxAllowedHits = cms.uint32( 50 ),
        performT0_vdriftSegCorrection = cms.bool( False ),
        AlphaMaxTheta = cms.double( 0.9 ),
        debug = cms.untracked.bool( False ),
        recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
        nUnSharedHitsMin = cms.int32( 2 ),
        performT0SegCorrection = cms.bool( False ),
        perform_delta_rejecting = cms.bool( False )
      ),
      performT0_vdriftSegCorrection = cms.bool( False ),
      debug = cms.untracked.bool( False ),
      recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
      nUnSharedHitsMin = cms.int32( 2 ),
      AllDTRecHits = cms.bool( True ),
      performT0SegCorrection = cms.bool( False ),
      perform_delta_rejecting = cms.bool( False )
    )
)
hltMuonCSCDigis = cms.EDProducer( "CSCDCCUnpacker",
    InputObjects = cms.InputTag( "rawDataCollector" ),
    UseExaminer = cms.bool( True ),
    ExaminerMask = cms.uint32( 0x1febf3f6 ),
    UseSelectiveUnpacking = cms.bool( True ),
    ErrorMask = cms.uint32( 0x0 ),
    UnpackStatusDigis = cms.bool( False ),
    UseFormatStatus = cms.bool( True ),
    PrintEventNumber = cms.untracked.bool( False )
)
hltCsc2DRecHits = cms.EDProducer( "CSCRecHitDProducer",
    CSCUseCalibrations = cms.bool( True ),
    CSCUseStaticPedestals = cms.bool( False ),
    CSCUseTimingCorrections = cms.bool( True ),
    CSCUseGasGainCorrections = cms.bool( False ),
    stripDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCStripDigi' ),
    wireDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCWireDigi' ),
    CSCstripWireDeltaTime = cms.int32( 8 ),
    CSCNoOfTimeBinsForDynamicPedestal = cms.int32( 2 ),
    CSCStripPeakThreshold = cms.double( 10.0 ),
    CSCStripClusterChargeCut = cms.double( 25.0 ),
    CSCWireClusterDeltaT = cms.int32( 1 ),
    CSCStripxtalksOffset = cms.double( 0.03 ),
    NoiseLevel_ME1a = cms.double( 7.0 ),
    XTasymmetry_ME1a = cms.double( 0.0 ),
    ConstSyst_ME1a = cms.double( 0.022 ),
    NoiseLevel_ME1b = cms.double( 8.0 ),
    XTasymmetry_ME1b = cms.double( 0.0 ),
    ConstSyst_ME1b = cms.double( 0.0070 ),
    NoiseLevel_ME12 = cms.double( 9.0 ),
    XTasymmetry_ME12 = cms.double( 0.0 ),
    ConstSyst_ME12 = cms.double( 0.0 ),
    NoiseLevel_ME13 = cms.double( 8.0 ),
    XTasymmetry_ME13 = cms.double( 0.0 ),
    ConstSyst_ME13 = cms.double( 0.0 ),
    NoiseLevel_ME21 = cms.double( 9.0 ),
    XTasymmetry_ME21 = cms.double( 0.0 ),
    ConstSyst_ME21 = cms.double( 0.0 ),
    NoiseLevel_ME22 = cms.double( 9.0 ),
    XTasymmetry_ME22 = cms.double( 0.0 ),
    ConstSyst_ME22 = cms.double( 0.0 ),
    NoiseLevel_ME31 = cms.double( 9.0 ),
    XTasymmetry_ME31 = cms.double( 0.0 ),
    ConstSyst_ME31 = cms.double( 0.0 ),
    NoiseLevel_ME32 = cms.double( 9.0 ),
    XTasymmetry_ME32 = cms.double( 0.0 ),
    ConstSyst_ME32 = cms.double( 0.0 ),
    NoiseLevel_ME41 = cms.double( 9.0 ),
    XTasymmetry_ME41 = cms.double( 0.0 ),
    ConstSyst_ME41 = cms.double( 0.0 ),
    readBadChannels = cms.bool( True ),
    readBadChambers = cms.bool( True ),
    UseAverageTime = cms.bool( False ),
    UseParabolaFit = cms.bool( False ),
    UseFivePoleFit = cms.bool( True )
)
hltCscSegments = cms.EDProducer( "CSCSegmentProducer",
    inputObjects = cms.InputTag( "hltCsc2DRecHits" ),
    algo_type = cms.int32( 1 ),
    algo_psets = cms.VPSet( 
      cms.PSet(  chamber_types = cms.vstring( 'ME1/a',
  'ME1/b',
  'ME1/2',
  'ME1/3',
  'ME2/1',
  'ME2/2',
  'ME3/1',
  'ME3/2',
  'ME4/1',
  'ME4/2' ),
        algo_name = cms.string( "CSCSegAlgoST" ),
        parameters_per_chamber_type = cms.vint32( 2, 1, 1, 1, 1, 1, 1, 1, 1, 1 ),
        algo_psets = cms.VPSet( 
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            yweightPenalty = cms.double( 1.5 ),
            maxRecHitsInCluster = cms.int32( 20 ),
            dPhiFineMax = cms.double( 0.025 ),
            preClusteringUseChaining = cms.bool( True ),
            ForceCovariance = cms.bool( False ),
            hitDropLimit6Hits = cms.double( 0.3333 ),
            NormChi2Cut2D = cms.double( 20.0 ),
            BPMinImprovement = cms.double( 10000.0 ),
            Covariance = cms.double( 0.0 ),
            tanPhiMax = cms.double( 0.5 ),
            SeedBig = cms.double( 0.0015 ),
            onlyBestSegment = cms.bool( False ),
            dRPhiFineMax = cms.double( 8.0 ),
            SeedSmall = cms.double( 2.0E-4 ),
            curvePenalty = cms.double( 2.0 ),
            dXclusBoxMax = cms.double( 4.0 ),
            BrutePruning = cms.bool( True ),
            curvePenaltyThreshold = cms.double( 0.85 ),
            CorrectTheErrors = cms.bool( True ),
            hitDropLimit4Hits = cms.double( 0.6 ),
            useShowering = cms.bool( False ),
            CSCDebug = cms.untracked.bool( False ),
            tanThetaMax = cms.double( 1.2 ),
            NormChi2Cut3D = cms.double( 10.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            ForceCovarianceAll = cms.bool( False ),
            yweightPenaltyThreshold = cms.double( 1.0 ),
            prePrunLimit = cms.double( 3.17 ),
            hitDropLimit5Hits = cms.double( 0.8 ),
            preClustering = cms.bool( True ),
            prePrun = cms.bool( True ),
            maxDPhi = cms.double( 999.0 ),
            maxDTheta = cms.double( 999.0 ),
            Pruning = cms.bool( True ),
            dYclusBoxMax = cms.double( 8.0 )
          ),
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            yweightPenalty = cms.double( 1.5 ),
            maxRecHitsInCluster = cms.int32( 24 ),
            dPhiFineMax = cms.double( 0.025 ),
            preClusteringUseChaining = cms.bool( True ),
            ForceCovariance = cms.bool( False ),
            hitDropLimit6Hits = cms.double( 0.3333 ),
            NormChi2Cut2D = cms.double( 20.0 ),
            BPMinImprovement = cms.double( 10000.0 ),
            Covariance = cms.double( 0.0 ),
            tanPhiMax = cms.double( 0.5 ),
            SeedBig = cms.double( 0.0015 ),
            onlyBestSegment = cms.bool( False ),
            dRPhiFineMax = cms.double( 8.0 ),
            SeedSmall = cms.double( 2.0E-4 ),
            curvePenalty = cms.double( 2.0 ),
            dXclusBoxMax = cms.double( 4.0 ),
            BrutePruning = cms.bool( True ),
            curvePenaltyThreshold = cms.double( 0.85 ),
            CorrectTheErrors = cms.bool( True ),
            hitDropLimit4Hits = cms.double( 0.6 ),
            useShowering = cms.bool( False ),
            CSCDebug = cms.untracked.bool( False ),
            tanThetaMax = cms.double( 1.2 ),
            NormChi2Cut3D = cms.double( 10.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            ForceCovarianceAll = cms.bool( False ),
            yweightPenaltyThreshold = cms.double( 1.0 ),
            prePrunLimit = cms.double( 3.17 ),
            hitDropLimit5Hits = cms.double( 0.8 ),
            preClustering = cms.bool( True ),
            prePrun = cms.bool( True ),
            maxDPhi = cms.double( 999.0 ),
            maxDTheta = cms.double( 999.0 ),
            Pruning = cms.bool( True ),
            dYclusBoxMax = cms.double( 8.0 )
          )
        )
      )
    )
)
hltMuonRPCDigis = cms.EDProducer( "RPCUnpackingModule",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    doSynchro = cms.bool( False )
)
hltRpcRecHits = cms.EDProducer( "RPCRecHitProducer",
    rpcDigiLabel = cms.InputTag( "hltMuonRPCDigis" ),
    recAlgo = cms.string( "RPCRecHitStandardAlgo" ),
    maskSource = cms.string( "File" ),
    maskvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat" ),
    deadSource = cms.string( "File" ),
    deadvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat" ),
    recAlgoConfig = cms.PSet(  )
)
hltL2MuonSeeds = cms.EDProducer( "L2MuonSeedGenerator",
    InputObjects = cms.InputTag( "hltL1extraParticles" ),
    GMTReadoutCollection = cms.InputTag( "hltGtDigis" ),
    Propagator = cms.string( "SteppingHelixPropagatorAny" ),
    L1MinPt = cms.double( 0.0 ),
    L1MaxEta = cms.double( 2.5 ),
    L1MinQuality = cms.uint32( 1 ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    )
)
hltL2Muons = cms.EDProducer( "L2MuonProducer",
    InputObjects = cms.InputTag( "hltL2MuonSeeds" ),
    L2TrajBuilderParameters = cms.PSet( 
      DoRefit = cms.bool( False ),
      SeedPropagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      FilterParameters = cms.PSet( 
        NumberOfSigma = cms.double( 3.0 ),
        FitDirection = cms.string( "insideOut" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        MaxChi2 = cms.double( 1000.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 25.0 ),
          RescaleErrorFactor = cms.double( 100.0 ),
          Granularity = cms.int32( 0 ),
          ExcludeRPCFromFit = cms.bool( False ),
          UseInvalidHits = cms.bool( True ),
          RescaleError = cms.bool( False )
        ),
        EnableRPCMeasurement = cms.bool( True ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        EnableDTMeasurement = cms.bool( True ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        EnableCSCMeasurement = cms.bool( True )
      ),
      NavigationType = cms.string( "Standard" ),
      SeedTransformerParameters = cms.PSet( 
        Fitter = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        NMinRecHits = cms.uint32( 2 ),
        UseSubRecHits = cms.bool( False ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        RescaleError = cms.double( 100.0 )
      ),
      DoBackwardFilter = cms.bool( True ),
      SeedPosition = cms.string( "in" ),
      BWFilterParameters = cms.PSet( 
        NumberOfSigma = cms.double( 3.0 ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        FitDirection = cms.string( "outsideIn" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        MaxChi2 = cms.double( 100.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 25.0 ),
          RescaleErrorFactor = cms.double( 100.0 ),
          Granularity = cms.int32( 2 ),
          ExcludeRPCFromFit = cms.bool( False ),
          UseInvalidHits = cms.bool( True ),
          RescaleError = cms.bool( False )
        ),
        EnableRPCMeasurement = cms.bool( True ),
        BWSeedType = cms.string( "fromGenerator" ),
        EnableDTMeasurement = cms.bool( True ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        EnableCSCMeasurement = cms.bool( True )
      ),
      DoSeedRefit = cms.bool( False )
    ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny',
        'hltESPFastSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    TrackLoaderParameters = cms.PSet( 
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
      DoSmoothing = cms.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        BeamSpotPosition = cms.vdouble( 0.0, 0.0, 0.0 ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( True )
    )
)
hltL2MuonCandidates = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
hltHIL2Mu3L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3withBptxANDL1Filtered" ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    saveTags = cms.bool( True ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 0 ),
    MinNhits = cms.vint32( 0 )
)
hltPreHIL2Mu3NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIL2Mu3NHitL2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3withBptxANDL1Filtered" ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    saveTags = cms.bool( True ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 0 ),
    MinNhits = cms.vint32( 1 )
)
hltPreHIL2Mu7 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIL2Mu7L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3withBptxANDL1Filtered" ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 ),
    saveTags = cms.bool( True ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 0 ),
    MinNhits = cms.vint32( 0 )
)
hltPreHIL2Mu15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIL2Mu15L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3withBptxANDL1Filtered" ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 ),
    saveTags = cms.bool( True ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 0 ),
    MinNhits = cms.vint32( 0 )
)
hltPreHIL2DoubleMu0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIL2DoubleMu0L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMuLevel1PathL1OpenFiltered" ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 3.0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 0.0 ),
    NSigmaPt = cms.double( 0.0 ),
    saveTags = cms.bool( True ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 0 ),
    MinNhits = cms.vint32( 0 )
)
hltPreHIL2DoubleMu0NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIL2DoubleMu0L2N1HitsFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMuLevel1PathL1OpenFiltered" ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 3.0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 0.0 ),
    NSigmaPt = cms.double( 0.0 ),
    saveTags = cms.bool( True ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 0 ),
    MinNhits = cms.vint32( 1 )
)
hltPreHIL2DoubleMu0L1HighQL2NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltPreHIL2DoubleMu3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIL2DoubleMu3L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMuLevel1PathL1OpenFiltered" ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 3.0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    saveTags = cms.bool( True ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 0 ),
    MinNhits = cms.vint32( 0 )
)
hltPreHIL3Mu3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltSiStripExcludedFEDListProducer = cms.EDProducer( "SiStripExcludedFEDListProducer",
    ProductLabel = cms.InputTag( "rawDataCollector" )
)
hltHISiStripRawToClustersFacility = cms.EDProducer( "SiStripRawToClusters",
    ProductLabel = cms.InputTag( "rawDataCollector" ),
    Clusterizer = cms.PSet( 
      ChannelThreshold = cms.double( 2.0 ),
      MaxSequentialBad = cms.uint32( 1 ),
      MaxSequentialHoles = cms.uint32( 0 ),
      Algorithm = cms.string( "ThreeThresholdAlgorithm" ),
      MaxAdjacentBad = cms.uint32( 0 ),
      QualityLabel = cms.string( "" ),
      SeedThreshold = cms.double( 3.0 ),
      ClusterThreshold = cms.double( 5.0 ),
      setDetId = cms.bool( True )
    ),
    Algorithms = cms.PSet( 
      SiStripFedZeroSuppressionMode = cms.uint32( 4 ),
      CommonModeNoiseSubtractionMode = cms.string( "IteratedMedian" ),
      PedestalSubtractionFedMode = cms.bool( False ),
      TruncateInSuppressor = cms.bool( True ),
      doAPVRestore = cms.bool( True ),
      useCMMeanMap = cms.bool( False ),
      CutToAvoidSignal = cms.double( 2.0 ),
      Iterations = cms.int32( 3 ),
      APVInspectMode = cms.string( "BaselineFollower" ),
      APVRestoreMode = cms.string( "BaselineFollower" ),
      ForceNoRestore = cms.bool( False ),
      SelfSelectRestoreAlgo = cms.bool( False ),
      useRealMeanCM = cms.bool( False ),
      DeltaCMThreshold = cms.uint32( 20 ),
      distortionThreshold = cms.uint32( 20 ),
      Fraction = cms.double( 0.2 ),
      Deviation = cms.uint32( 25 ),
      restoreThreshold = cms.double( 0.5 ),
      nSaturatedStrip = cms.uint32( 2 ),
      nSigmaNoiseDerTh = cms.uint32( 4 ),
      consecThreshold = cms.uint32( 5 ),
      hitStripThreshold = cms.uint32( 40 ),
      nSmooth = cms.uint32( 9 ),
      minStripsToFit = cms.uint32( 4 ),
      ApplyBaselineCleaner = cms.bool( True ),
      MeanCM = cms.int32( 0 ),
      slopeX = cms.int32( 3 ),
      slopeY = cms.int32( 4 ),
      ApplyBaselineRejection = cms.bool( True ),
      filteredBaselineMax = cms.double( 6.0 ),
      filteredBaselineDerivativeSumSquare = cms.double( 30.0 ),
      CleaningSequence = cms.uint32( 1 )
    )
)
hltHISiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltHISiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" )
)
hltL3TrajSeedOIState = cms.EDProducer( "TSGFromL2Muon",
    PtCut = cms.double( 1.0 ),
    PCut = cms.double( 2.5 ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSteppingHelixPropagatorOpposite',
        'hltESPSteppingHelixPropagatorAlong' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    MuonTrackingRegionBuilder = cms.PSet(  ),
    TkSeedGenerator = cms.PSet( 
      propagatorCompatibleName = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
      option = cms.uint32( 3 ),
      maxChi2 = cms.double( 40.0 ),
      errorMatrixPset = cms.PSet( 
        atIP = cms.bool( True ),
        action = cms.string( "use" ),
        errorMatrixValuesPSet = cms.PSet( 
          pf3_V12 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V13 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V11 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V14 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V15 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          yAxis = cms.vdouble( 0.0, 1.0, 1.4, 10.0 ),
          pf3_V33 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          zAxis = cms.vdouble( -3.14159, 3.14159 ),
          pf3_V44 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          xAxis = cms.vdouble( 0.0, 13.0, 30.0, 70.0, 1000.0 ),
          pf3_V22 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V23 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V45 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V55 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V34 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V35 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V25 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V24 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          )
        )
      ),
      propagatorName = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
      manySeeds = cms.bool( False ),
      copyMuonRecHit = cms.bool( False ),
      ComponentName = cms.string( "TSGForRoadSearch" )
    ),
    TrackerSeedCleaner = cms.PSet(  ),
    TSGFromMixedPairs = cms.PSet(  ),
    TSGFromPixelTriplets = cms.PSet(  ),
    TSGFromPixelPairs = cms.PSet(  ),
    TSGForRoadSearchOI = cms.PSet(  ),
    TSGForRoadSearchIOpxl = cms.PSet(  ),
    TSGFromPropagation = cms.PSet(  ),
    TSGFromCombinedHits = cms.PSet(  )
)
hltHIL3TrackCandidateFromL2OIState = cms.EDProducer( "CkfTrajectoryMaker",
    trackCandidateAlso = cms.bool( True ),
    src = cms.InputTag( "hltL3TrajSeedOIState" ),
    TrajectoryBuilder = cms.string( "hltHIAllESPMuonCkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    doSeedingRegionRebuilding = cms.bool( False ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 )
)
hltHIL3TkTracksFromL2OIState = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    Propagator = cms.string( "PropagatorWithMaterial" ),
    src = cms.InputTag( "hltHIL3TrackCandidateFromL2OIState" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    NavigationSchool = cms.string( "" )
)
hltHIL3MuonsOIState = cms.EDProducer( "L3MuonProducer",
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    L3TrajBuilderParameters = cms.PSet( 
      ScaleTECyFactor = cms.double( -1.0 ),
      GlbRefitterParameters = cms.PSet( 
        TrackerSkipSection = cms.int32( -1 ),
        DoPredictionsOnly = cms.bool( False ),
        PropDirForCosmics = cms.bool( False ),
        HitThreshold = cms.int32( 1 ),
        MuonHitsOption = cms.int32( 1 ),
        Chi2CutRPC = cms.double( 1.0 ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        Chi2CutCSC = cms.double( 150.0 ),
        Chi2CutDT = cms.double( 10.0 ),
        RefitRPCHits = cms.bool( True ),
        SkipStation = cms.int32( -1 ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        TrackerSkipSystem = cms.int32( -1 ),
        DYTthrs = cms.vint32( 30, 15 )
      ),
      ScaleTECxFactor = cms.double( -1.0 ),
      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      MuonTrackingRegionBuilder = cms.PSet( 
        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
        OnDemand = cms.double( -1.0 ),
        Rescale_Dz = cms.double( 3.0 ),
        vertexCollection = cms.InputTag( "pixelVertices" ),
        Rescale_phi = cms.double( 3.0 ),
        Eta_fixed = cms.double( 0.2 ),
        DeltaZ_Region = cms.double( 15.9 ),
        MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
        Eta_min = cms.double( 0.05 ),
        Phi_fixed = cms.double( 0.2 ),
        DeltaR = cms.double( 0.2 ),
        EscapePt = cms.double( 1.5 ),
        UseFixedRegion = cms.bool( False ),
        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
        Rescale_eta = cms.double( 3.0 ),
        Phi_min = cms.double( 0.05 ),
        UseVertex = cms.bool( False ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
      ),
      RefitRPCHits = cms.bool( True ),
      PCut = cms.double( 2.5 ),
      TrackTransformer = cms.PSet( 
        DoPredictionsOnly = cms.bool( False ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" )
      ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Pt_threshold1 = cms.double( 0.0 ),
        DeltaDCut_3 = cms.double( 15.0 ),
        MinP = cms.double( 2.5 ),
        MinPt = cms.double( 1.0 ),
        Chi2Cut_1 = cms.double( 50.0 ),
        Pt_threshold2 = cms.double( 9.99999999E8 ),
        LocChi2Cut = cms.double( 0.0010 ),
        Eta_threshold = cms.double( 1.2 ),
        Quality_3 = cms.double( 7.0 ),
        Quality_2 = cms.double( 15.0 ),
        Chi2Cut_2 = cms.double( 50.0 ),
        Chi2Cut_3 = cms.double( 200.0 ),
        DeltaDCut_1 = cms.double( 40.0 ),
        DeltaRCut_2 = cms.double( 0.2 ),
        DeltaRCut_3 = cms.double( 1.0 ),
        DeltaDCut_2 = cms.double( 10.0 ),
        DeltaRCut_1 = cms.double( 0.1 ),
        Propagator = cms.string( "hltESPSmartPropagator" ),
        Quality_1 = cms.double( 20.0 )
      ),
      PtCut = cms.double( 1.0 ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      tkTrajLabel = cms.InputTag( "hltHIL3TkTracksFromL2OIState" )
    ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    TrackLoaderParameters = cms.PSet( 
      PutTkTrackIntoEvent = cms.untracked.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      SmoothTkTrack = cms.untracked.bool( False ),
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( False ),
      DoSmoothing = cms.bool( True )
    )
)
hltHIL3TrajSeedOIHit = cms.EDProducer( "TSGFromL2Muon",
    PtCut = cms.double( 1.0 ),
    PCut = cms.double( 2.5 ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'PropagatorWithMaterial',
        'hltESPSmartPropagatorAnyOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    MuonTrackingRegionBuilder = cms.PSet(  ),
    TkSeedGenerator = cms.PSet( 
      PSetNames = cms.vstring( 'skipTSG',
        'iterativeTSG' ),
      L3TkCollectionA = cms.InputTag( "hltHIL3MuonsOIState" ),
      iterativeTSG = cms.PSet( 
        ErrorRescaling = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "offlineBeamSpot" ),
        MaxChi2 = cms.double( 40.0 ),
        errorMatrixPset = cms.PSet( 
          atIP = cms.bool( True ),
          action = cms.string( "use" ),
          errorMatrixValuesPSet = cms.PSet( 
            pf3_V12 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V13 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V11 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            pf3_V14 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V15 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            yAxis = cms.vdouble( 0.0, 1.0, 1.4, 10.0 ),
            pf3_V33 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            zAxis = cms.vdouble( -3.14159, 3.14159 ),
            pf3_V44 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            xAxis = cms.vdouble( 0.0, 13.0, 30.0, 70.0, 1000.0 ),
            pf3_V22 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            pf3_V23 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V45 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V55 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            pf3_V34 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V35 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V25 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V24 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            )
          )
        ),
        UpdateState = cms.bool( True ),
        MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
        SelectState = cms.bool( False ),
        SigmaZ = cms.double( 25.0 ),
        ResetMethod = cms.string( "matrix" ),
        ComponentName = cms.string( "TSGFromPropagation" ),
        UseVertexState = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" )
      ),
      skipTSG = cms.PSet(  ),
      ComponentName = cms.string( "DualByL2TSG" )
    ),
    TrackerSeedCleaner = cms.PSet( 
      cleanerFromSharedHits = cms.bool( True ),
      ptCleaner = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      directionCleaner = cms.bool( True )
    ),
    TSGFromMixedPairs = cms.PSet(  ),
    TSGFromPixelTriplets = cms.PSet(  ),
    TSGFromPixelPairs = cms.PSet(  ),
    TSGForRoadSearchOI = cms.PSet(  ),
    TSGForRoadSearchIOpxl = cms.PSet(  ),
    TSGFromPropagation = cms.PSet(  ),
    TSGFromCombinedHits = cms.PSet(  )
)
hltHIL3TrackCandidateFromL2OIHit = cms.EDProducer( "CkfTrajectoryMaker",
    trackCandidateAlso = cms.bool( True ),
    src = cms.InputTag( "hltHIL3TrajSeedOIHit" ),
    TrajectoryBuilder = cms.string( "hltHIAllESPMuonCkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    doSeedingRegionRebuilding = cms.bool( False ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 )
)
hltHIL3TkTracksFromL2OIHit = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    Propagator = cms.string( "PropagatorWithMaterial" ),
    src = cms.InputTag( "hltHIL3TrackCandidateFromL2OIHit" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    NavigationSchool = cms.string( "" )
)
hltHIL3MuonsOIHit = cms.EDProducer( "L3MuonProducer",
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    L3TrajBuilderParameters = cms.PSet( 
      ScaleTECyFactor = cms.double( -1.0 ),
      GlbRefitterParameters = cms.PSet( 
        TrackerSkipSection = cms.int32( -1 ),
        DoPredictionsOnly = cms.bool( False ),
        PropDirForCosmics = cms.bool( False ),
        HitThreshold = cms.int32( 1 ),
        MuonHitsOption = cms.int32( 1 ),
        Chi2CutRPC = cms.double( 1.0 ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        Chi2CutCSC = cms.double( 150.0 ),
        Chi2CutDT = cms.double( 10.0 ),
        RefitRPCHits = cms.bool( True ),
        SkipStation = cms.int32( -1 ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        TrackerSkipSystem = cms.int32( -1 ),
        DYTthrs = cms.vint32( 30, 15 )
      ),
      ScaleTECxFactor = cms.double( -1.0 ),
      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      MuonTrackingRegionBuilder = cms.PSet( 
        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
        OnDemand = cms.double( -1.0 ),
        Rescale_Dz = cms.double( 3.0 ),
        vertexCollection = cms.InputTag( "pixelVertices" ),
        Rescale_phi = cms.double( 3.0 ),
        Eta_fixed = cms.double( 0.2 ),
        DeltaZ_Region = cms.double( 15.9 ),
        MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
        Eta_min = cms.double( 0.05 ),
        Phi_fixed = cms.double( 0.2 ),
        DeltaR = cms.double( 0.2 ),
        EscapePt = cms.double( 1.5 ),
        UseFixedRegion = cms.bool( False ),
        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
        Rescale_eta = cms.double( 3.0 ),
        Phi_min = cms.double( 0.05 ),
        UseVertex = cms.bool( False ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
      ),
      RefitRPCHits = cms.bool( True ),
      PCut = cms.double( 2.5 ),
      TrackTransformer = cms.PSet( 
        DoPredictionsOnly = cms.bool( False ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" )
      ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Pt_threshold1 = cms.double( 0.0 ),
        DeltaDCut_3 = cms.double( 15.0 ),
        MinP = cms.double( 2.5 ),
        MinPt = cms.double( 1.0 ),
        Chi2Cut_1 = cms.double( 50.0 ),
        Pt_threshold2 = cms.double( 9.99999999E8 ),
        LocChi2Cut = cms.double( 0.0010 ),
        Eta_threshold = cms.double( 1.2 ),
        Quality_3 = cms.double( 7.0 ),
        Quality_2 = cms.double( 15.0 ),
        Chi2Cut_2 = cms.double( 50.0 ),
        Chi2Cut_3 = cms.double( 200.0 ),
        DeltaDCut_1 = cms.double( 40.0 ),
        DeltaRCut_2 = cms.double( 0.2 ),
        DeltaRCut_3 = cms.double( 1.0 ),
        DeltaDCut_2 = cms.double( 10.0 ),
        DeltaRCut_1 = cms.double( 0.1 ),
        Propagator = cms.string( "hltESPSmartPropagator" ),
        Quality_1 = cms.double( 20.0 )
      ),
      PtCut = cms.double( 1.0 ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      tkTrajLabel = cms.InputTag( "hltHIL3TkTracksFromL2OIHit" )
    ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    TrackLoaderParameters = cms.PSet( 
      PutTkTrackIntoEvent = cms.untracked.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      SmoothTkTrack = cms.untracked.bool( False ),
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( False ),
      DoSmoothing = cms.bool( True )
    )
)
hltHIL3TkFromL2OICombination = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit' )
)
hltHIL3TrajSeedIOHit = cms.EDProducer( "TSGFromL2Muon",
    PtCut = cms.double( 1.0 ),
    PCut = cms.double( 2.5 ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'PropagatorWithMaterial' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    MuonTrackingRegionBuilder = cms.PSet( 
      EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
      EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
      OnDemand = cms.double( -1.0 ),
      Rescale_Dz = cms.double( 3.0 ),
      vertexCollection = cms.InputTag( "pixelVertices" ),
      Rescale_phi = cms.double( 3.0 ),
      Eta_fixed = cms.double( 0.2 ),
      DeltaZ_Region = cms.double( 15.9 ),
      MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
      PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
      Eta_min = cms.double( 0.1 ),
      Phi_fixed = cms.double( 0.2 ),
      DeltaR = cms.double( 0.2 ),
      EscapePt = cms.double( 1.5 ),
      UseFixedRegion = cms.bool( False ),
      PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
      Rescale_eta = cms.double( 3.0 ),
      Phi_min = cms.double( 0.1 ),
      UseVertex = cms.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
    ),
    TkSeedGenerator = cms.PSet( 
      PSetNames = cms.vstring( 'skipTSG',
        'skipTSG' ),
      L3TkCollectionA = cms.InputTag( "hltHIL3TkFromL2OICombination" ),
      iterativeTSG = cms.PSet( 
        firstTSG = cms.PSet( 
          ComponentName = cms.string( "TSGFromOrderedHits" ),
          OrderedHitsFactoryPSet = cms.PSet( 
            ComponentName = cms.string( "StandardHitTripletGenerator" ),
            GeneratorPSet = cms.PSet( 
              useBending = cms.bool( True ),
              useFixedPreFiltering = cms.bool( False ),
              maxElement = cms.uint32( 0 ),
              phiPreFiltering = cms.double( 0.3 ),
              extraHitRPhitolerance = cms.double( 0.06 ),
              useMultScattering = cms.bool( True ),
              ComponentName = cms.string( "PixelTripletHLTGenerator" ),
              extraHitRZtolerance = cms.double( 0.06 ),
              SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
            ),
            SeedingLayers = cms.string( "hltESPPixelLayerTriplets" )
          ),
          TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
        ),
        PSetNames = cms.vstring( 'firstTSG',
          'secondTSG' ),
        ComponentName = cms.string( "CombinedTSG" ),
        thirdTSG = cms.PSet( 
          PSetNames = cms.vstring( 'endcapTSG',
            'barrelTSG' ),
          barrelTSG = cms.PSet(  ),
          endcapTSG = cms.PSet( 
            ComponentName = cms.string( "TSGFromOrderedHits" ),
            OrderedHitsFactoryPSet = cms.PSet( 
              maxElement = cms.uint32( 0 ),
              ComponentName = cms.string( "StandardHitPairGenerator" ),
              SeedingLayers = cms.string( "hltESPMixedLayerPairs" ),
              useOnDemandTracker = cms.untracked.int32( 0 )
            ),
            TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
          ),
          etaSeparation = cms.double( 2.0 ),
          ComponentName = cms.string( "DualByEtaTSG" )
        ),
        secondTSG = cms.PSet( 
          ComponentName = cms.string( "TSGFromOrderedHits" ),
          OrderedHitsFactoryPSet = cms.PSet( 
            maxElement = cms.uint32( 0 ),
            ComponentName = cms.string( "StandardHitPairGenerator" ),
            SeedingLayers = cms.string( "hltESPPixelLayerPairs" ),
            useOnDemandTracker = cms.untracked.int32( 0 )
          ),
          TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
        )
      ),
      skipTSG = cms.PSet(  ),
      ComponentName = cms.string( "DualByL2TSG" )
    ),
    TrackerSeedCleaner = cms.PSet( 
      cleanerFromSharedHits = cms.bool( True ),
      ptCleaner = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      directionCleaner = cms.bool( True )
    ),
    TSGFromMixedPairs = cms.PSet(  ),
    TSGFromPixelTriplets = cms.PSet(  ),
    TSGFromPixelPairs = cms.PSet(  ),
    TSGForRoadSearchOI = cms.PSet(  ),
    TSGForRoadSearchIOpxl = cms.PSet(  ),
    TSGFromPropagation = cms.PSet(  ),
    TSGFromCombinedHits = cms.PSet(  )
)
hltHIL3TrackCandidateFromL2IOHit = cms.EDProducer( "CkfTrajectoryMaker",
    trackCandidateAlso = cms.bool( True ),
    src = cms.InputTag( "hltHIL3TrajSeedIOHit" ),
    TrajectoryBuilder = cms.string( "hltHIAllESPMuonCkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    doSeedingRegionRebuilding = cms.bool( False ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 )
)
hltHIL3TkTracksFromL2IOHit = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    Propagator = cms.string( "PropagatorWithMaterial" ),
    src = cms.InputTag( "hltHIL3TrackCandidateFromL2IOHit" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    NavigationSchool = cms.string( "" )
)
hltHIAllL3MuonsIOHit = cms.EDProducer( "L3MuonProducer",
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    L3TrajBuilderParameters = cms.PSet( 
      ScaleTECyFactor = cms.double( -1.0 ),
      GlbRefitterParameters = cms.PSet( 
        TrackerSkipSection = cms.int32( -1 ),
        DoPredictionsOnly = cms.bool( False ),
        PropDirForCosmics = cms.bool( False ),
        HitThreshold = cms.int32( 1 ),
        MuonHitsOption = cms.int32( 1 ),
        Chi2CutRPC = cms.double( 1.0 ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        Chi2CutCSC = cms.double( 150.0 ),
        Chi2CutDT = cms.double( 10.0 ),
        RefitRPCHits = cms.bool( True ),
        SkipStation = cms.int32( -1 ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        TrackerSkipSystem = cms.int32( -1 ),
        DYTthrs = cms.vint32( 30, 15 )
      ),
      ScaleTECxFactor = cms.double( -1.0 ),
      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      MuonTrackingRegionBuilder = cms.PSet( 
        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
        OnDemand = cms.double( -1.0 ),
        Rescale_Dz = cms.double( 3.0 ),
        vertexCollection = cms.InputTag( "pixelVertices" ),
        Rescale_phi = cms.double( 3.0 ),
        Eta_fixed = cms.double( 0.2 ),
        DeltaZ_Region = cms.double( 15.9 ),
        MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
        Eta_min = cms.double( 0.05 ),
        Phi_fixed = cms.double( 0.2 ),
        DeltaR = cms.double( 0.2 ),
        EscapePt = cms.double( 1.5 ),
        UseFixedRegion = cms.bool( False ),
        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
        Rescale_eta = cms.double( 3.0 ),
        Phi_min = cms.double( 0.05 ),
        UseVertex = cms.bool( False ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
      ),
      RefitRPCHits = cms.bool( True ),
      PCut = cms.double( 2.5 ),
      TrackTransformer = cms.PSet( 
        DoPredictionsOnly = cms.bool( False ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" )
      ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Pt_threshold1 = cms.double( 0.0 ),
        DeltaDCut_3 = cms.double( 15.0 ),
        MinP = cms.double( 2.5 ),
        MinPt = cms.double( 1.0 ),
        Chi2Cut_1 = cms.double( 50.0 ),
        Pt_threshold2 = cms.double( 9.99999999E8 ),
        LocChi2Cut = cms.double( 0.0010 ),
        Eta_threshold = cms.double( 1.2 ),
        Quality_3 = cms.double( 7.0 ),
        Quality_2 = cms.double( 15.0 ),
        Chi2Cut_2 = cms.double( 50.0 ),
        Chi2Cut_3 = cms.double( 200.0 ),
        DeltaDCut_1 = cms.double( 40.0 ),
        DeltaRCut_2 = cms.double( 0.2 ),
        DeltaRCut_3 = cms.double( 1.0 ),
        DeltaDCut_2 = cms.double( 10.0 ),
        DeltaRCut_1 = cms.double( 0.1 ),
        Propagator = cms.string( "hltESPSmartPropagator" ),
        Quality_1 = cms.double( 20.0 )
      ),
      PtCut = cms.double( 1.0 ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      tkTrajLabel = cms.InputTag( "hltHIL3TkTracksFromL2IOHit" )
    ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    TrackLoaderParameters = cms.PSet( 
      PutTkTrackIntoEvent = cms.untracked.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      SmoothTkTrack = cms.untracked.bool( False ),
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( False ),
      DoSmoothing = cms.bool( True )
    )
)
hltHIL3TrajectorySeed = cms.EDProducer( "L3MuonTrajectorySeedCombiner",
    labels = cms.VInputTag( 'hltHIL3TrajSeedIOHit','hltL3TrajSeedOIState','hltHIL3TrajSeedOIHit' )
)
hltHIL3TrackCandidateFromL2 = cms.EDProducer( "L3TrackCandCombiner",
    labels = cms.VInputTag( 'hltHIL3TrackCandidateFromL2IOHit','hltHIL3TrackCandidateFromL2OIHit','hltHIL3TrackCandidateFromL2OIState' )
)
hltHIL3TkTracksFromL2 = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltHIL3TkTracksFromL2IOHit','hltHIL3TkTracksFromL2OIHit','hltHIL3TkTracksFromL2OIState' )
)
hltHIL3MuonsLinksCombination = cms.EDProducer( "L3TrackLinksCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit','hltHIAllL3MuonsIOHit' )
)
hltHIL3Muons = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit','hltHIAllL3MuonsIOHit' )
)
hltHIL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputObjects = cms.InputTag( "hltHIL3Muons" ),
    InputLinksObjects = cms.InputTag( "hltHIL3MuonsLinksCombination" ),
    MuonPtOption = cms.string( "Global" )
)
hltHISingleMu3L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIL2Mu3L2Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    saveTags = cms.bool( True )
)
hltPreHIL1DoubleMuOpenL3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDimuonL2PreFiltered0 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMuLevel1PathL1OpenFiltered" ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 0.0 ),
    NSigmaPt = cms.double( 0.0 ),
    saveTags = cms.bool( True ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 0 ),
    MinNhits = cms.vint32( 0 )
)
hltHIDimuonL3FilteredOpen = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonL2PreFiltered0" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 20.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 0.0 ),
    MinPtMin = cms.double( 0.0 ),
    MinInvMass = cms.double( 0.0 ),
    MaxInvMass = cms.double( 300.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    saveTags = cms.bool( True ),
    CutCowboys = cms.bool( False )
)
hltPreHIL1DoubleMuOpenL3Mgt2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDimuonL3FilteredMgt2 = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonL2PreFiltered0" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 20.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 0.0 ),
    MinPtMin = cms.double( 0.0 ),
    MinInvMass = cms.double( 2.0 ),
    MaxInvMass = cms.double( 300.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    saveTags = cms.bool( True ),
    CutCowboys = cms.bool( False )
)
hltPreHIL1DoubleMuOpenL3Mgt2SS = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDimuonL3FilteredMgt2SS = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonL2PreFiltered0" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 20.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 1 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 0.0 ),
    MinPtMin = cms.double( 0.0 ),
    MinInvMass = cms.double( 2.0 ),
    MaxInvMass = cms.double( 300.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    saveTags = cms.bool( True ),
    CutCowboys = cms.bool( False )
)
hltPreHIL1DoubleMuOpenL3Mgt2OS = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDimuonL3FilteredMg2OS = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonL2PreFiltered0" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 20.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( -1 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 0.0 ),
    MinPtMin = cms.double( 0.0 ),
    MinInvMass = cms.double( 2.0 ),
    MaxInvMass = cms.double( 300.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    saveTags = cms.bool( True ),
    CutCowboys = cms.bool( False )
)
hltPreHIL1DoubleMuOpenL3Mgt2OSnoCowboy = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDimuonL3FilteredMg2OSnoCowboy = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonL2PreFiltered0" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 20.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( -1 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 0.0 ),
    MinPtMin = cms.double( 0.0 ),
    MinInvMass = cms.double( 2.0 ),
    MaxInvMass = cms.double( 300.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    saveTags = cms.bool( True ),
    CutCowboys = cms.bool( True )
)
hltL1sL1SingleEG5BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5_BptxAND OR L1_ETT2000" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIPhoton15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltIslandBasicClustersHI = cms.EDProducer( "IslandClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    barrelHitProducer = cms.string( "hltEcalRecHitAll" ),
    endcapHitProducer = cms.string( "hltEcalRecHitAll" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClustersHI" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClustersHI" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    clustershapecollectionEB = cms.string( "islandBarrelShape" ),
    clustershapecollectionEE = cms.string( "islandEndcapShape" ),
    barrelShapeAssociation = cms.string( "islandBarrelShapeAssoc" ),
    endcapShapeAssociation = cms.string( "islandEndcapShapeAssoc" ),
    posCalcParameters = cms.PSet( 
      T0_barl = cms.double( 7.4 ),
      LogWeighted = cms.bool( True ),
      T0_endc = cms.double( 3.1 ),
      T0_endcPresh = cms.double( 1.2 ),
      W0 = cms.double( 4.2 ),
      X0 = cms.double( 0.89 )
    )
)
hltHiIslandSuperClustersHI = cms.EDProducer( "HiSuperClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    endcapClusterProducer = cms.string( "hltIslandBasicClustersHI" ),
    barrelClusterProducer = cms.string( "hltIslandBasicClustersHI" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClustersHI" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClustersHI" ),
    endcapSuperclusterCollection = cms.string( "islandEndcapSuperClustersHI" ),
    barrelSuperclusterCollection = cms.string( "islandBarrelSuperClustersHI" ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( True ),
    barrelEtaSearchRoad = cms.double( 0.07 ),
    barrelPhiSearchRoad = cms.double( 0.8 ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    endcapPhiSearchRoad = cms.double( 0.6 ),
    seedTransverseEnergyThreshold = cms.double( 1.0 ),
    barrelBCEnergyThreshold = cms.double( 0.0 ),
    endcapBCEnergyThreshold = cms.double( 0.0 )
)
hltHiCorrectedIslandBarrelSuperClustersHI = cms.EDProducer( "HiEgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( 'hltHiIslandSuperClustersHI','islandBarrelSuperClustersHI' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      fEtaVect = cms.vdouble( 0.993, 0.0, 0.00546, 1.165, -0.180844, 0.040312 ),
      fBremVect = cms.vdouble( -0.773799, 2.73438, -1.07235, 0.986821, -0.0101822, 3.06744E-4, 1.00595, -0.0495958, 0.00451986, 1.00595, -0.0495958, 0.00451986 ),
      fBremThVect = cms.vdouble( 1.2, 1.2 ),
      fEtEtaVect = cms.vdouble( 0.9497, 0.006985, 1.03754, -0.0142667, -0.0233993, 0.0, 0.0, 0.908915, 0.0137322, 16.9602, -29.3093, 19.8976, -5.92666, 0.654571 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 ),
      minR9Barrel = cms.double( 0.94 ),
      minR9Endcap = cms.double( 0.95 ),
      maxR9 = cms.double( 1.5 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltHiCorrectedIslandEndcapSuperClustersHI = cms.EDProducer( "HiEgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    rawSuperClusterProducer = cms.InputTag( 'hltHiIslandSuperClustersHI','islandEndcapSuperClustersHI' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      fEtaVect = cms.vdouble( 0.993, 0.0, 0.00546, 1.165, -0.180844, 0.040312 ),
      fBremVect = cms.vdouble( -0.773799, 2.73438, -1.07235, 0.986821, -0.0101822, 3.06744E-4, 1.00595, -0.0495958, 0.00451986, 1.00595, -0.0495958, 0.00451986 ),
      fBremThVect = cms.vdouble( 1.2, 1.2 ),
      fEtEtaVect = cms.vdouble( 0.9497, 0.006985, 1.03754, -0.0142667, -0.0233993, 0.0, 0.0, 0.908915, 0.0137322, 16.9602, -29.3093, 19.8976, -5.92666, 0.654571 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 ),
      minR9Barrel = cms.double( 0.94 ),
      minR9Endcap = cms.double( 0.95 ),
      maxR9 = cms.double( 1.5 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCleanedHiCorrectedIslandBarrelSuperClustersHI = cms.EDProducer( "HiSpikeCleaner",
    recHitProducerBarrel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    recHitProducerEndcap = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    originalSuperClusterProducer = cms.InputTag( "hltHiCorrectedIslandBarrelSuperClustersHI" ),
    TimingCut = cms.untracked.double( 9999999.0 ),
    etCut = cms.double( 8.0 ),
    outputColl = cms.string( "" )
)
hltRecoHIEcalWithCleaningCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag( "hltCleanedHiCorrectedIslandBarrelSuperClustersHI" ),
    scIslandEndcapProducer = cms.InputTag( "hltHiCorrectedIslandEndcapSuperClustersHI" ),
    recoEcalCandidateCollection = cms.string( "" )
)
hltHIPhoton15 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 15.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 1 )
)
hltPreHIPhoton20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIPhoton20 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 20.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 1 )
)
hltPreHIPhoton30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIPhoton30 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 1 )
)
hltPreHIPhoton40 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIPhoton40 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 1 )
)
hltL1sL1DoubleEG5BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG5_BptxAND OR L1_ETT2000" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIDoublePhoton10and15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDoublePhoton1015Filter1 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 10.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 2 )
)
hltHIDoublePhoton1015Filter2 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 15.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 1 )
)
hltPreHIDoublePhoton15and20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDoublePhoton1520Filter1 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 15.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 2 )
)
hltHIDoublePhoton1520Filter2 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 20.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 1 )
)
hltPreHIDoublePhoton10 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDoublePhoton10 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 10.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 2 )
)
hltPreHIDoublePhoton15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDoublePhoton15 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 15.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 2 )
)
hltPreHIDoublePhoton20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDoublePhoton20 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 20.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 2 )
)
hltL1sL1SingleJet36BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36_BptxAND OR L1_ETT2000" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIJet55 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltIterativeCone5PileupSubtractionCaloJets = cms.EDProducer( "FastjetJetProducer",
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    DzTrVtxMax = cms.double( 0.0 ),
    DxyTrVtxMax = cms.double( 0.0 ),
    MinVtxNdof = cms.int32( 5 ),
    MaxVtxZ = cms.double( 15.0 ),
    jetAlgorithm = cms.string( "IterativeCone" ),
    rParam = cms.double( 0.5 ),
    src = cms.InputTag( "hltTowerMakerForAll" ),
    srcPVs = cms.InputTag( "offlinePrimaryVertices" ),
    jetType = cms.string( "CaloJet" ),
    jetPtMin = cms.double( 10.0 ),
    inputEtMin = cms.double( 0.3 ),
    inputEMin = cms.double( 0.0 ),
    doPVCorrection = cms.bool( False ),
    doPUOffsetCorr = cms.bool( True ),
    nSigmaPU = cms.double( 1.0 ),
    radiusPU = cms.double( 0.5 ),
    Active_Area_Repeats = cms.int32( 5 ),
    GhostArea = cms.double( 0.01 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doAreaFastjet = cms.bool( False ),
    doRhoFastjet = cms.bool( False ),
    subtractorName = cms.string( "MultipleAlgoIterator" ),
    sumRecHits = cms.bool( False ),
    doAreaDiskApprox = cms.bool( False ),
    voronoiRfact = cms.double( -9.0 ),
    useDeterministicSeed = cms.bool( False ),
    minSeed = cms.uint32( 0 ),
    Rho_EtaMax = cms.double( 4.4 ),
    puPtMin = cms.double( 10.0 )
)
hltHICaloJetIDPassed = cms.EDProducer( "HLTJetIDProducer",
    jetsInput = cms.InputTag( "hltIterativeCone5PileupSubtractionCaloJets" ),
    min_EMF = cms.double( 1.0E-6 ),
    max_EMF = cms.double( 999.0 ),
    min_N90 = cms.int32( -2 ),
    min_N90hits = cms.int32( -2 ),
    JetIDParams = cms.PSet( 
      useRecHits = cms.bool( True ),
      hbheRecHitsColl = cms.InputTag( "hltHbhereco" ),
      hoRecHitsColl = cms.InputTag( "hltHoreco" ),
      hfRecHitsColl = cms.InputTag( "hltHfreco" ),
      ebRecHitsColl = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
      eeRecHitsColl = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' )
    )
)
hltHICaloJetCorrected = cms.EDProducer( "CaloJetCorrectionProducer",
    src = cms.InputTag( "hltHICaloJetIDPassed" ),
    verbose = cms.untracked.bool( False ),
    alias = cms.untracked.string( "JetCorJetIC5" ),
    correctors = cms.vstring( 'hltESSAK5CaloL2L3' )
)
hltHI1jet55 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltHICaloJetCorrected" ),
    saveTags = cms.bool( True ),
    MinPt = cms.double( 55.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltPreHIJet65 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHI1jet65 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltHICaloJetCorrected" ),
    saveTags = cms.bool( True ),
    MinPt = cms.double( 65.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1sL1SingleJet52BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet52_BptxAND OR L1_ETT2000" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIJet80 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHI1jet80 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltHICaloJetCorrected" ),
    saveTags = cms.bool( True ),
    MinPt = cms.double( 80.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1sL1SingleJet68BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet52_BptxAND OR L1_ETT2000" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIJet95 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHI1jet95 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltHICaloJetCorrected" ),
    saveTags = cms.bool( True ),
    MinPt = cms.double( 95.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltPreHIDiJet55and55 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIDiJet55and55 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltHICaloJetCorrected" ),
    saveTags = cms.bool( True ),
    MinPt = cms.double( 55.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltPreHIDiJet65and55 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sL1SingleJet20CentralNotBptx = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet20_Central_NotBptxOR_NotMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreJetE30NoBPTX = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltStoppedHSCPTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.07 ),
    EEThreshold = cms.double( 0.3 ),
    UseEtEBTreshold = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    UseSymEBTreshold = cms.bool( False ),
    UseSymEETreshold = cms.bool( False ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.7 ),
    HESThreshold = cms.double( 0.8 ),
    HEDThreshold = cms.double( 0.8 ),
    HOThreshold0 = cms.double( 3.5 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HF1Threshold = cms.double( 0.5 ),
    HF2Threshold = cms.double( 0.85 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0E-99 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHBDepth = cms.double( 0.2 ),
    MomHEDepth = cms.double( 0.4 ),
    MomEBDepth = cms.double( 0.3 ),
    MomEEDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "" ),
    hfInput = cms.InputTag( "" ),
    AllowMissingInputs = cms.bool( True ),
    HcalAcceptSeverityLevel = cms.uint32( 9 ),
    UseHcalRecoveredHits = cms.bool( False ),
    UseEcalRecoveredHits = cms.bool( False ),
    UseRejectedHitsOnly = cms.bool( False ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    UseRejectedRecoveredHcalHits = cms.bool( False ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    EBGrid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    EEWeights = cms.vdouble(  ),
    HBGrid = cms.vdouble(  ),
    HBWeights = cms.vdouble(  ),
    HESGrid = cms.vdouble(  ),
    HESWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HOGrid = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    HF1Grid = cms.vdouble(  ),
    HF1Weights = cms.vdouble(  ),
    HF2Grid = cms.vdouble(  ),
    HF2Weights = cms.vdouble(  ),
    ecalInputs = cms.VInputTag(  ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kTime',
      'kWeird',
      'kBad' ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  )
)
hltStoppedHSCPIterativeCone5CaloJets = cms.EDProducer( "FastjetJetProducer",
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    DzTrVtxMax = cms.double( 0.0 ),
    DxyTrVtxMax = cms.double( 0.0 ),
    MinVtxNdof = cms.int32( 5 ),
    MaxVtxZ = cms.double( 15.0 ),
    jetAlgorithm = cms.string( "IterativeCone" ),
    rParam = cms.double( 0.5 ),
    src = cms.InputTag( "hltStoppedHSCPTowerMakerForAll" ),
    srcPVs = cms.InputTag( "offlinePrimaryVertices" ),
    jetType = cms.string( "CaloJet" ),
    jetPtMin = cms.double( 1.0 ),
    inputEtMin = cms.double( 0.3 ),
    inputEMin = cms.double( 0.0 ),
    doPVCorrection = cms.bool( False ),
    doPUOffsetCorr = cms.bool( False ),
    nSigmaPU = cms.double( 1.0 ),
    radiusPU = cms.double( 0.5 ),
    Active_Area_Repeats = cms.int32( 5 ),
    GhostArea = cms.double( 0.01 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doAreaFastjet = cms.bool( False ),
    doRhoFastjet = cms.bool( False ),
    subtractorName = cms.string( "" ),
    sumRecHits = cms.bool( False ),
    doAreaDiskApprox = cms.bool( False ),
    voronoiRfact = cms.double( -9.0 ),
    useDeterministicSeed = cms.bool( False ),
    minSeed = cms.uint32( 0 ),
    Rho_EtaMax = cms.double( 4.4 ),
    puPtMin = cms.double( 10.0 )
)
hltStoppedHSCPControl1CaloJetEnergy30 = cms.EDFilter( "HLT1CaloJetEnergy",
    inputTag = cms.InputTag( "hltStoppedHSCPIterativeCone5CaloJets" ),
    saveTags = cms.bool( True ),
    MinE = cms.double( 30.0 ),
    MaxEta = cms.double( 3.0 ),
    MinN = cms.int32( 1 )
)
hltBPTXAntiCoincidence = cms.EDFilter( "HLTLevel1Activity",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    daqPartitions = cms.uint32( 1 ),
    ignoreL1Mask = cms.bool( True ),
    invert = cms.bool( True ),
    physicsLoBits = cms.uint64( 0x1 ),
    physicsHiBits = cms.uint64( 0x0 ),
    technicalBits = cms.uint64( 0x11 ),
    bunchCrossings = cms.vint32( 0, 1, -1 )
)
hltL1sL1SingleJet32CentralNotBptx = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet32_Central_NotBptxOR_NotMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltL1BeamHaloAntiCoincidence3BX = cms.EDFilter( "HLTLevel1Activity",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    daqPartitions = cms.uint32( 1 ),
    ignoreL1Mask = cms.bool( True ),
    invert = cms.bool( True ),
    physicsLoBits = cms.uint64( 0x40000000000000 ),
    physicsHiBits = cms.uint64( 0x0 ),
    technicalBits = cms.uint64( 0x0 ),
    bunchCrossings = cms.vint32( 0, 1, -1 )
)
hltPreJetE50NoBPTX3BXNoHalo = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltStoppedHSCPHpdFilter = cms.EDFilter( "HLTHPDFilter",
    inputTag = cms.InputTag( "hltHbhereco" ),
    energy = cms.double( -99.0 ),
    hpdSpikeEnergy = cms.double( 10.0 ),
    hpdSpikeIsolationEnergy = cms.double( 1.0 ),
    rbxSpikeEnergy = cms.double( 50.0 ),
    rbxSpikeUnbalance = cms.double( 0.2 )
)
hltStoppedHSCPCaloJetEnergy50 = cms.EDFilter( "HLT1CaloJetEnergy",
    inputTag = cms.InputTag( "hltStoppedHSCPIterativeCone5CaloJets" ),
    saveTags = cms.bool( True ),
    MinE = cms.double( 50.0 ),
    MaxEta = cms.double( 3.0 ),
    MinN = cms.int32( 1 )
)
hltL1sL1GlobalDecision = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1GlobalDecision" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIActivityHFCoincidence3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHcalSimpleRecHitFilterCoincidence = cms.EDFilter( "HLTHcalSimpleRecHitFilter",
    threshold = cms.double( 3.0 ),
    minNHitsNeg = cms.int32( 1 ),
    minNHitsPos = cms.int32( 1 ),
    doCoincidence = cms.bool( True ),
    HFRecHitCollection = cms.InputTag( "hltHfreco" ),
    maskedChannels = cms.vint32(  )
)
hltPreHIActivityHFSingle3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHcalSimpleRecHitFilter = cms.EDFilter( "HLTHcalSimpleRecHitFilter",
    threshold = cms.double( 3.0 ),
    minNHitsNeg = cms.int32( 1 ),
    minNHitsPos = cms.int32( 1 ),
    doCoincidence = cms.bool( False ),
    HFRecHitCollection = cms.InputTag( "hltHfreco" ),
    maskedChannels = cms.vint32(  )
)
hltPreHIClusterVertexCompatibility = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHIPixelClusterShapeFilter = cms.EDFilter( "HLTPixelClusterShapeFilter",
    inputTag = cms.InputTag( "hltHISiPixelRecHits" ),
    saveTags = cms.bool( False ),
    minZ = cms.double( -20.0 ),
    maxZ = cms.double( 20.05 ),
    zStep = cms.double( 0.2 ),
    nhitsTrunc = cms.int32( 150 ),
    clusterTrunc = cms.double( 2.0 ),
    clusterPars = cms.vdouble(  )
)
hltPreHICentralityVeto = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltPixelActivityFilterCentralityVeto = cms.EDFilter( "HLTPixelActivityFilter",
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    saveTags = cms.bool( False ),
    minClusters = cms.uint32( 3 ),
    maxClusters = cms.uint32( 1000 )
)
hltBPTXCoincidence = cms.EDFilter( "HLTLevel1Activity",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    daqPartitions = cms.uint32( 1 ),
    ignoreL1Mask = cms.bool( True ),
    invert = cms.bool( False ),
    physicsLoBits = cms.uint64( 0x1 ),
    physicsHiBits = cms.uint64( 0x40000 ),
    technicalBits = cms.uint64( 0x1 ),
    bunchCrossings = cms.vint32( 0, -1, 1 )
)
hltL1sETT100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETT100_BptxAND OR L1_ETT2000" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIFullTrack12L1Central = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHICaloTowerFilter4 = cms.EDFilter( "HLTCaloTowerFilter",
    inputTag = cms.InputTag( "hltTowerMakerForAll" ),
    saveTags = cms.bool( False ),
    MinPt = cms.double( 4.0 ),
    MaxEta = cms.double( 2.4 ),
    MinN = cms.uint32( 1 )
)
hltHIPixelClusterVerticesForHITrackTrigger = cms.EDProducer( "HIPixelClusterVtxProducer",
    pixelRecHits = cms.string( "hltHISiPixelRecHits" ),
    minZ = cms.double( -15.0 ),
    maxZ = cms.double( 15.0 ),
    zStep = cms.double( 1.0 )
)
hltHIPixel3ProtoTracks = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( False ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "HITrackingRegionForPrimaryVtxProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.1 ),
        ptMin = cms.double( 1.0 ),
        directionXCoord = cms.double( 1.0 ),
        directionZCoord = cms.double( 0.0 ),
        directionYCoord = cms.double( 1.0 ),
        useFoundVertices = cms.bool( True ),
        doVariablePtMin = cms.bool( True ),
        nSigmaZ = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFixedError = cms.bool( True ),
        fixedError = cms.double( 1.0 ),
        sigmaZVertex = cms.double( 3.0 ),
        siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" ),
        VertexCollection = cms.InputTag( "hltHIPixelClusterVerticesForHITrackTrigger" )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      SeedingLayers = cms.string( "hltESPHIPixelLayerTriplets" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.037 ),
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      )
    ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
    ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIProtoTrackFilter" ),
      ptMin = cms.double( 1.0 ),
      tipMax = cms.double( 1.0 ),
      doVariablePtMin = cms.bool( True ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) )
)
hltHIPixelMedianVertex = cms.EDProducer( "HIPixelMedianVtxProducer",
    TrackCollection = cms.InputTag( "hltHIPixel3ProtoTracks" ),
    PtMin = cms.double( 0.075 ),
    PeakFindThreshold = cms.uint32( 100 ),
    PeakFindMaxZ = cms.double( 30.0 ),
    PeakFindBinsPerCm = cms.int32( 10 ),
    FitThreshold = cms.int32( 5 ),
    FitMaxZ = cms.double( 0.1 ),
    FitBinsPerCm = cms.int32( 500 )
)
hltHISelectedProtoTracks = cms.EDFilter( "HIProtoTrackSelection",
    src = cms.InputTag( "hltHIPixel3ProtoTracks" ),
    VertexCollection = cms.InputTag( "hltHIPixelMedianVertex" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    ptMin = cms.double( 0.0 ),
    nSigmaZ = cms.double( 5.0 ),
    minZCut = cms.double( 0.2 ),
    maxD0Significance = cms.double( 5.0 )
)
hltHIPixelAdaptiveVertex = cms.EDProducer( "PrimaryVertexProducer",
    TrackLabel = cms.InputTag( "hltHISelectedProtoTracks" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    algorithm = cms.string( "AdaptiveVertexFitter" ),
    label = cms.string( "" ),
    minNdof = cms.double( 0.0 ),
    useBeamConstraint = cms.bool( False ),
    maxDistanceToBeam = cms.double( 2.0 ),
    TkFilterParameters = cms.PSet( 
      maxD0Significance = cms.double( 3.0 ),
      minPt = cms.double( 0.0 ),
      maxNormalizedChi2 = cms.double( 5.0 ),
      minSiliconLayersWithHits = cms.int32( 0 ),
      minPixelLayersWithHits = cms.int32( 2 ),
      trackQuality = cms.string( "any" ),
      numTracksThreshold = cms.int32( 2 ),
      algorithm = cms.string( "filterWithThreshold" )
    ),
    PVSelParameters = cms.PSet(  maxDistanceToBeam = cms.double( 0.1 ) ),
    TkClusParameters = cms.PSet( 
      algorithm = cms.string( "gap" ),
      TkGapClusParameters = cms.PSet(  zSeparation = cms.double( 1.0 ) )
    ),
    vertexCollections = cms.VPSet( 
      cms.PSet(  maxDistanceToBeam = cms.double( 2.0 ),
        useBeamConstraint = cms.bool( False ),
        minNdof = cms.double( 0.0 ),
        algorithm = cms.string( "AdaptiveVertexFitter" ),
        label = cms.string( "" )
      )
    )
)
hltHIBestAdaptiveVertex = cms.EDFilter( "HIBestVertexSelection",
    src = cms.InputTag( "hltHIPixelAdaptiveVertex" ),
    maxNumber = cms.uint32( 1 )
)
hltHISelectedVertex = cms.EDProducer( "HIBestVertexProducer",
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    medianVertexCollection = cms.InputTag( "hltHIPixelMedianVertex" ),
    adaptiveVertexCollection = cms.InputTag( "hltHIBestAdaptiveVertex" )
)
hltHIPixel3PrimTracks = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( True ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "GlobalTrackingRegionWithVerticesProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.05 ),
        ptMin = cms.double( 0.9 ),
        useFoundVertices = cms.bool( True ),
        nSigmaZ = cms.double( 2.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFixedError = cms.bool( True ),
        fixedError = cms.double( 0.2 ),
        sigmaZVertex = cms.double( 3.0 ),
        VertexCollection = cms.InputTag( "hltHISelectedVertex" )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      SeedingLayers = cms.string( "hltESPHIPixelLayerTriplets" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.037 ),
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      )
    ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
    ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIPixelTrackFilter" ),
      ptMin = cms.double( 0.9 ),
      tipMax = cms.double( 0.0 ),
      useClusterShape = cms.bool( False ),
      VertexCollection = cms.InputTag( "hltHISelectedVertex" ),
      nSigmaTipMaxTolerance = cms.double( 6.0 ),
      nSigmaLipMaxTolerance = cms.double( 0.0 ),
      lipMax = cms.double( 0.3 )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "TrackCleaner" ) )
)
hltHIPixelTrackCandsForHITrackTrigger = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIPixel3PrimTracks" ),
    particleType = cms.string( "pi+" )
)
hltHISinglePixelTrackFilter10 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    trackCollection = cms.InputTag( "hltHIPixelTrackCandsForHITrackTrigger" ),
    MinPt = cms.double( 10.0 ),
    MaxPt = cms.double( 10000.0 ),
    MaxEta = cms.double( 2.4 ),
    MaxVz = cms.double( 15.0 ),
    MinTrks = cms.int32( 1 ),
    MinSep = cms.double( 0.2 )
)
hltHISiStripClustersNonRegional = cms.EDProducer( "SiStripClusterizer",
    Clusterizer = cms.PSet( 
      ChannelThreshold = cms.double( 2.0 ),
      MaxSequentialBad = cms.uint32( 1 ),
      Algorithm = cms.string( "ThreeThresholdAlgorithm" ),
      MaxSequentialHoles = cms.uint32( 0 ),
      MaxAdjacentBad = cms.uint32( 0 ),
      QualityLabel = cms.string( "" ),
      SeedThreshold = cms.double( 3.0 ),
      ClusterThreshold = cms.double( 5.0 )
    ),
    DigiProducersList = cms.VInputTag( 'hltSiStripZeroSuppression:VirginRaw','hltSiStripZeroSuppression:ProcessedRaw','hltSiStripZeroSuppression:ScopeMode' )
)
hltHIPixelTrackSeeds = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    InputCollection = cms.InputTag( "hltHIPixel3PrimTracks" ),
    InputVertexCollection = cms.InputTag( "" ),
    originHalfLength = cms.double( 1.0E9 ),
    originRadius = cms.double( 1.0E9 ),
    useProtoTrackKinematics = cms.bool( False ),
    useEventsWithNoVertex = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
)
hltHIPrimTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltHIPixelTrackSeeds" ),
    TrajectoryBuilder = cms.string( "hltESPCkfTrajectoryBuilderForHI" ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedSeeds" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "none" ),
    useHitsSplitting = cms.bool( True ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialForHI" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOppositeForHI" ),
      numberMeasurementsForFit = cms.int32( 4 )
    ),
    cleanTrajectoryAfterInOut = cms.bool( True ),
    maxNSeeds = cms.uint32( 100000 )
)
hltHIGlobalPrimTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( True ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltHIPrimTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
    AlgorithmName = cms.string( "PropagatorWithMaterialOppositeForHI" ),
    NavigationSchool = cms.string( "" )
)
hltHIGoodLooseTracks = cms.EDProducer( "AnalyticalTrackSelector",
    src = cms.InputTag( "hltHIGlobalPrimTracks" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    useVertices = cms.bool( True ),
    useVtxError = cms.bool( True ),
    vertices = cms.InputTag( "hltHISelectedVertex" ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    vertexCut = cms.string( "" ),
    chi2n_par = cms.double( 9999.0 ),
    chi2n_no1Dmod_par = cms.double( 0.2 ),
    applyAdaptedPVCuts = cms.bool( True ),
    max_d0 = cms.double( 100.0 ),
    max_z0 = cms.double( 100.0 ),
    nSigmaZ = cms.double( 4.0 ),
    minNumberLayers = cms.uint32( 6 ),
    minNumber3DLayers = cms.uint32( 0 ),
    maxNumberLostLayers = cms.uint32( 999 ),
    max_relpterr = cms.double( 0.055 ),
    min_nhits = cms.uint32( 11 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    keepAllTracks = cms.bool( False ),
    qualityBit = cms.string( "loose" ),
    max_d0NoPV = cms.double( 0.2 ),
    max_z0NoPV = cms.double( 15.0 ),
    res_par = cms.vdouble( 99999.0, 99999.0 ),
    d0_par1 = cms.vdouble( 9999.0, 0.0 ),
    dz_par1 = cms.vdouble( 9999.0, 0.0 ),
    d0_par2 = cms.vdouble( 5.0, 0.0 ),
    dz_par2 = cms.vdouble( 5.0, 0.0 )
)
hltHIFullTrackCandsForHITrackTrigger = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIGoodLooseTracks" ),
    particleType = cms.string( "pi+" )
)
hltHISingleFullTrackFilter12 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    MinPt = cms.double( 12.0 ),
    MaxPt = cms.double( 10000.0 ),
    MaxEta = cms.double( 2.4 ),
    MaxVz = cms.double( 15.0 ),
    MinTrks = cms.int32( 1 ),
    MinSep = cms.double( 0.2 )
)
hltL1sJet16CentralNotETT140BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet16_Central_NotETT140_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIFullTrack12L1Peripheral = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltPreHIFullTrack14L1Central = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHISingleFullTrackFilter14 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    MinPt = cms.double( 14.0 ),
    MaxPt = cms.double( 10000.0 ),
    MaxEta = cms.double( 2.4 ),
    MaxVz = cms.double( 15.0 ),
    MinTrks = cms.int32( 1 ),
    MinSep = cms.double( 0.2 )
)
hltPreHIFullTrack14L1Peripheral = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltPreHIFullTrack20L1Central = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHISingleFullTrackFilter20 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    MinPt = cms.double( 20.0 ),
    MaxPt = cms.double( 10000.0 ),
    MaxEta = cms.double( 2.4 ),
    MaxVz = cms.double( 15.0 ),
    MinTrks = cms.int32( 1 ),
    MinSep = cms.double( 0.2 )
)
hltPreHIFullTrack20L1Peripheral = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltPreHIFullTrack25L1Central = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltHISingleFullTrackFilter25 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    MinPt = cms.double( 25.0 ),
    MaxPt = cms.double( 10000.0 ),
    MaxEta = cms.double( 2.4 ),
    MaxVz = cms.double( 15.0 ),
    MinTrks = cms.int32( 1 ),
    MinSep = cms.double( 0.2 )
)
hltPreHIFullTrack25L1Peripheral = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltRandomEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 3 )
)
hltPreHIRandom = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sETT2000 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETT2000" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIUCC010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltTowerMakerForHf = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.07 ),
    EEThreshold = cms.double( 0.3 ),
    UseEtEBTreshold = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    UseSymEBTreshold = cms.bool( False ),
    UseSymEETreshold = cms.bool( False ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.7 ),
    HESThreshold = cms.double( 0.8 ),
    HEDThreshold = cms.double( 0.8 ),
    HOThreshold0 = cms.double( 3.5 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HF1Threshold = cms.double( 0.5 ),
    HF2Threshold = cms.double( 0.85 ),
    EBWeight = cms.double( 1.0E-99 ),
    EEWeight = cms.double( 1.0E-99 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0E-99 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHBDepth = cms.double( 0.2 ),
    MomHEDepth = cms.double( 0.4 ),
    MomEBDepth = cms.double( 0.3 ),
    MomEEDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "" ),
    hoInput = cms.InputTag( "" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    AllowMissingInputs = cms.bool( True ),
    HcalAcceptSeverityLevel = cms.uint32( 11 ),
    UseHcalRecoveredHits = cms.bool( True ),
    UseEcalRecoveredHits = cms.bool( True ),
    UseRejectedHitsOnly = cms.bool( False ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    UseRejectedRecoveredHcalHits = cms.bool( True ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    EBGrid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    EEWeights = cms.vdouble(  ),
    HBGrid = cms.vdouble(  ),
    HBWeights = cms.vdouble(  ),
    HESGrid = cms.vdouble(  ),
    HESWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HOGrid = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    HF1Grid = cms.vdouble(  ),
    HF1Weights = cms.vdouble(  ),
    HF2Grid = cms.vdouble(  ),
    HF2Weights = cms.vdouble(  ),
    ecalInputs = cms.VInputTag(  ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kProblematic',
      'kRecovered',
      'kTime',
      'kWeird',
      'kBad' ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  )
)
hltMetForHf = cms.EDProducer( "METProducer",
    src = cms.InputTag( "hltTowerMakerForHf" ),
    InputType = cms.string( "CandidateCollection" ),
    METType = cms.string( "CaloMET" ),
    alias = cms.string( "RawCaloMET" ),
    globalThreshold = cms.double( 0.5 ),
    noHF = cms.bool( False ),
    calculateSignificance = cms.bool( False ),
    onlyFiducialParticles = cms.bool( False ),
    usePt = cms.untracked.bool( True ),
    jets = cms.InputTag( "" ),
    rf_type = cms.int32( 0 ),
    correctShowerTracks = cms.bool( False ),
    HO_EtResPar = cms.vdouble( 0.0, 1.3, 0.0050 ),
    HF_EtResPar = cms.vdouble( 0.0, 1.82, 0.09 ),
    HB_PhiResPar = cms.vdouble( 0.02511 ),
    HE_PhiResPar = cms.vdouble( 0.02511 ),
    EE_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    EB_PhiResPar = cms.vdouble( 0.00502 ),
    EE_PhiResPar = cms.vdouble( 0.02511 ),
    HB_EtResPar = cms.vdouble( 0.0, 1.22, 0.05 ),
    EB_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    HF_PhiResPar = cms.vdouble( 0.05022 ),
    HE_EtResPar = cms.vdouble( 0.0, 1.3, 0.05 ),
    HO_PhiResPar = cms.vdouble( 0.02511 )
)
hltGlobalSumETHfFilter3200 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    inputTag = cms.InputTag( "hltMetForHf" ),
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 3200.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPixelActivityFilter51500 = cms.EDFilter( "HLTPixelActivityFilter",
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    saveTags = cms.bool( False ),
    minClusters = cms.uint32( 51500 ),
    maxClusters = cms.uint32( 1000000 )
)
hltPreHIUCC015 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltGlobalSumETHfFilter3100 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    inputTag = cms.InputTag( "hltMetForHf" ),
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 3100.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPixelActivityFilter51400 = cms.EDFilter( "HLTPixelActivityFilter",
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    saveTags = cms.bool( False ),
    minClusters = cms.uint32( 51400 ),
    maxClusters = cms.uint32( 1000000 )
)
hltPreHICentral10 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltGlobalSumETHfFilter0 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    inputTag = cms.InputTag( "hltMetForHf" ),
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 0.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPixelActivityFilter0 = cms.EDFilter( "HLTPixelActivityFilter",
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    saveTags = cms.bool( False ),
    minClusters = cms.uint32( 34000 ),
    maxClusters = cms.uint32( 1000000 )
)
hltL1sHIUPCNeuMu = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_MuOpen_ZdcCalo_NotBscMinBiasThresh2_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIUPCNeuMuPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIUPCNeuEG2 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_EG2_ZdcCalo_NotBscMinBiasThresh2_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIUPCNeuEG2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIUPCNeuEG5 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_EG5_ZdcCalo_NotBscMinBiasThresh2_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIUPCNeuEG5PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIUPCNeuHcalHfMu = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_MuOpen_ZdcCalo_NotHcalHfCoincidencePm_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIUPCNeuHcalHfMuPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIUPCNeuHcalHfEG2 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_EG2_ZdcCalo_NotHcalHfCoincidencePm_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIUPCNeuHcalHfEG2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sHIUPCNeuHcalHfEG5 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_EG5_ZdcCalo_NotHcalHfCoincidencePm_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.bool( True )
)
hltPreHIUPCNeuHcalHfEG5PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
hltFEDSelector = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1023 )
)
hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
)
hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
hltL1GtTrigReport = cms.EDAnalyzer( "L1GtTrigReport",
    UseL1GlobalTriggerRecord = cms.bool( False ),
    L1GtRecordInputTag = cms.InputTag( "hltGtDigis" )
)
hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)

HLTL1UnpackerSequence = cms.Sequence( hltGtDigis + hltGctDigis + hltL1GtObjectMap + hltL1extraParticles )
HLTBeamSpot = cms.Sequence( hltScalersRawToDigi + hltOnlineBeamSpot + hltOfflineBeamSpot )
HLTBeginSequence = cms.Sequence( hltTriggerType + HLTL1UnpackerSequence + HLTBeamSpot )
HLTDoLocalHcalSequence = cms.Sequence( hltHcalDigis + hltHbhereco + hltHfreco + hltHoreco )
HLTDoCaloSequence = cms.Sequence( hltEcalRawToRecHitFacility + hltEcalRegionalRestFEDs + hltEcalRecHitAll + HLTDoLocalHcalSequence + hltTowerMakerForAll )
HLTRecoMETSequence = cms.Sequence( HLTDoCaloSequence + hltMet )
HLTDoHIStripZeroSuppression = cms.Sequence( hltSiStripRawToDigi + hltSiStripZeroSuppression + hltSiStripDigiToZSRaw + hltSiStripRawDigiToVirginRaw + virginRawDataRepacker + rawDataRepacker )
HLTEndSequence = cms.Sequence( hltBoolEnd )
HLTBeginSequenceCalibration = cms.Sequence( hltCalibrationEventsFilter + hltGtDigis )
HLTDoHILocalPixelSequence = cms.Sequence( hltSiPixelDigis + hltHISiPixelClusters + hltHISiPixelRecHits )
HLTPixelTrackingForHITrackTrigger = cms.Sequence( hltHIPixelClusterVertices + hltPixelTracksForHITrackTrigger + hltPixelCandsForHITrackTrigger )
HLTMuonLocalRecoSequence = cms.Sequence( hltMuonDTDigis + hltDt1DRecHits + hltDt4DSegments + hltMuonCSCDigis + hltCsc2DRecHits + hltCscSegments + hltMuonRPCDigis + hltRpcRecHits )
HLTL2muonrecoNocandSequence = cms.Sequence( HLTMuonLocalRecoSequence + hltL2MuonSeeds + hltL2Muons )
HLTL2muonrecoSequence = cms.Sequence( HLTL2muonrecoNocandSequence + hltL2MuonCandidates )
HLTDoHILocalStripSequence = cms.Sequence( hltSiStripExcludedFEDListProducer + hltHISiStripRawToClustersFacility + hltHISiStripClusters )
HLTHIL3muonTkCandidateSequence = cms.Sequence( HLTDoHILocalPixelSequence + HLTDoHILocalStripSequence + hltL3TrajSeedOIState + hltHIL3TrackCandidateFromL2OIState + hltHIL3TkTracksFromL2OIState + hltHIL3MuonsOIState + hltHIL3TrajSeedOIHit + hltHIL3TrackCandidateFromL2OIHit + hltHIL3TkTracksFromL2OIHit + hltHIL3MuonsOIHit + hltHIL3TkFromL2OICombination + hltHIL3TrajSeedIOHit + hltHIL3TrackCandidateFromL2IOHit + hltHIL3TkTracksFromL2IOHit + hltHIAllL3MuonsIOHit + hltHIL3TrajectorySeed + hltHIL3TrackCandidateFromL2 )
HLTHIL3muonrecoNocandSequence = cms.Sequence( HLTHIL3muonTkCandidateSequence + hltHIL3TkTracksFromL2 + hltHIL3MuonsLinksCombination + hltHIL3Muons )
HLTHIL3muonrecoSequence = cms.Sequence( HLTHIL3muonrecoNocandSequence + hltHIL3MuonCandidates )
HLTDoHIEcalClusWithCleaningSequence = cms.Sequence( hltIslandBasicClustersHI + hltHiIslandSuperClustersHI + hltHiCorrectedIslandBarrelSuperClustersHI + hltHiCorrectedIslandEndcapSuperClustersHI + hltCleanedHiCorrectedIslandBarrelSuperClustersHI + hltRecoHIEcalWithCleaningCandidate )
HLTHIRecoJetSequenceIC5Uncorrected = cms.Sequence( HLTDoCaloSequence + hltIterativeCone5PileupSubtractionCaloJets )
HLTHIRecoJetSequenceIC5Corrected = cms.Sequence( HLTHIRecoJetSequenceIC5Uncorrected + hltHICaloJetIDPassed + hltHICaloJetCorrected )
HLTStoppedHSCPLocalHcalReco = cms.Sequence( hltHcalDigis + hltHbhereco )
HLTStoppedHSCPJetSequence = cms.Sequence( hltStoppedHSCPTowerMakerForAll + hltStoppedHSCPIterativeCone5CaloJets )
HLTBeginSequenceAntiBPTX = cms.Sequence( hltTriggerType + HLTL1UnpackerSequence + hltBPTXAntiCoincidence + HLTBeamSpot )
HLTBeginSequenceBPTX = cms.Sequence( hltTriggerType + HLTL1UnpackerSequence + hltBPTXCoincidence + HLTBeamSpot )
HLTPixelSeedingForHITrackTrigger = cms.Sequence( hltHIPixelClusterVerticesForHITrackTrigger + hltHIPixel3ProtoTracks + hltHIPixelMedianVertex + hltHISelectedProtoTracks + hltHIPixelAdaptiveVertex + hltHIBestAdaptiveVertex + hltHISelectedVertex + hltHIPixel3PrimTracks + hltHIPixelTrackCandsForHITrackTrigger )
HLTDoHILocalStripSequenceNonRegional = cms.Sequence( hltSiStripRawToDigi + hltSiStripZeroSuppression + hltHISiStripClustersNonRegional )
HLTFullTrackingForHITrackTrigger = cms.Sequence( hltHIPixelTrackSeeds + hltHIPrimTrackCandidates + hltHIGlobalPrimTracks + hltHIGoodLooseTracks + hltHIFullTrackCandsForHITrackTrigger )
HLTDoLocalHfSequence = cms.Sequence( hltHcalDigis + hltHfreco + hltTowerMakerForHf )
HLTRecoMETHfSequence = cms.Sequence( HLTDoLocalHfSequence + hltMetForHf )
HLTDoHILocalPixelClustersSequence = cms.Sequence( hltSiPixelDigis + hltHISiPixelClusters )

HLTriggerFirstPath = cms.Path( hltGetRaw + hltBoolFalse )
HLT_HIMET120_v1 = cms.Path( HLTBeginSequence + hltL1sL1ETM30BptxAND + hltPreHIMET120 + HLTRecoMETSequence + hltHIMET120 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMET200_v1 = cms.Path( HLTBeginSequence + hltL1sL1ETM50BptxAND + hltPreHIMET200 + HLTRecoMETSequence + hltHIMET200 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMET220_v1 = cms.Path( HLTBeginSequence + hltL1sL1ETM50BptxAND + hltPreHIMET220 + HLTRecoMETSequence + hltHIMET220 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIPhysics_v1 = cms.Path( HLTBeginSequence + hltPreHIPhysics + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIDTCalibration_v1 = cms.Path( HLTBeginSequenceCalibration + hltPreHIDTCalibration + hltDTCalibrationRaw + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIEcalCalibration_v1 = cms.Path( HLTBeginSequenceCalibration + hltPreHIEcalCalibration + hltEcalCalibrationRaw + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIHcalCalibration_v1 = cms.Path( HLTBeginSequenceCalibration + hltPreHIHcalCalibration + hltHcalCalibTypeFilter + hltHcalCalibrationRaw + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIZeroBias_v1 = cms.Path( HLTBeginSequence + hltL1sHIZeroBias + hltPreHIZeroBias + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIZeroBiasXOR_v1 = cms.Path( HLTBeginSequence + hltL1sL1BptxXOR + hltPreHIZeroBiasXOR + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIZeroBiasPixel_SingleTrack_v1 = cms.Path( HLTBeginSequence + hltL1sHIZeroBiasXOR + hltPreHIZeroBiasPixelSingleTrack + HLTDoHILocalPixelSequence + HLTPixelTrackingForHITrackTrigger + hltHISinglePixelTrackFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMinBiasBSC_v1 = cms.Path( HLTBeginSequence + hltL1sHIMinBiasBSC + hltPreHIMinBiasBSC + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMinBiasBSC_OR_v1 = cms.Path( HLTBeginSequence + hltL1sHIMinBiasBSCOR + hltPreHIMinBiasBSCOR + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMinBiasHF_v1 = cms.Path( HLTBeginSequence + hltL1sHIMinBiasHF + hltPreHIMinBiasHF + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMinBiasHf_OR_v1 = cms.Path( HLTBeginSequence + hltL1sHIMinBiasHfOr + hltPreHIMinBiasHfOR + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMinBiasHfOrBSC_v1 = cms.Path( HLTBeginSequence + hltL1sHIMinBiasHfOrBSC + hltPreHIMinBiasHfOrBSC + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMinBiasPixel_SingleTrack_v1 = cms.Path( HLTBeginSequence + hltL1sHIMinBiasHfOrBSC + hltPreHIMinBiasPixelSingleTrack + HLTDoHILocalPixelSequence + HLTPixelTrackingForHITrackTrigger + hltHISinglePixelTrackFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMinBiasZDC_Calo_v1 = cms.Path( HLTBeginSequence + hltL1sHIMinBiasZDC + hltPreHIMinBiasZDCCalo + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMinBiasZDC_Calo_PlusOrMinus_v1 = cms.Path( HLTBeginSequence + hltL1sHIMinBiasZDCCaloPlusOrMinus + hltPreHIMinBiasZDCCaloPlusOrMinus + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMinBiasZDCPixel_SingleTrack_v1 = cms.Path( HLTBeginSequence + hltL1sHIMinBiasZDCPixelSingleTrack + hltPreHIMinBiasZDCPixelSingleTrack + HLTDoHILocalPixelSequence + HLTPixelTrackingForHITrackTrigger + hltHISinglePixelTrackFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v1 = cms.Path( HLTBeginSequence + hltL1sHIMinBiasZDCPlusOrMinusPixelSingleTrack + hltPreHIMinBiasZDCPlusOrMinusPixelSingleTrack + HLTDoHILocalPixelSequence + HLTPixelTrackingForHITrackTrigger + hltHISinglePixelTrackFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIBptxXOR_v1 = cms.Path( HLTBeginSequence + hltL1sL1BptxXOR + hltPreHIBptxXOR + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL1Algo_BptxXOR_BSC_OR_v1 = cms.Path( HLTBeginSequence + hltL1sL1BptxXORBscMinBiasOR + hltPreHIL1AlgoBptxXORBSCOR + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL1DoubleMuOpen_v2 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpenBptxAND + hltPreHIL1DoubleMuOpen + hltHIDoubleMuLevel1PathL1OpenFiltered + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL1DoubleMu0_HighQ_v2 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpenBptxAND + hltPreHIL1DoubleMuOpenHighQ + hltHIDoubleMuLevel1PathL1HighQFiltered + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL2Mu3_v2 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu3BptxAND + hltPreHIL2Mu3 + hltHIL1SingleMu3withBptxANDL1Filtered + HLTL2muonrecoSequence + hltHIL2Mu3L2Filtered + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL2Mu3_NHitQ_v2 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu3BptxAND + hltPreHIL2Mu3NHitQ + hltHIL1SingleMu3withBptxANDL1Filtered + HLTL2muonrecoSequence + hltHIL2Mu3NHitL2Filtered + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL2Mu7_v2 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu3BptxAND + hltPreHIL2Mu7 + hltHIL1SingleMu3withBptxANDL1Filtered + HLTL2muonrecoSequence + hltHIL2Mu7L2Filtered + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL2Mu15_v2 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu3BptxAND + hltPreHIL2Mu15 + hltHIL1SingleMu3withBptxANDL1Filtered + HLTL2muonrecoSequence + hltHIL2Mu15L2Filtered + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL2DoubleMu0_v2 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpenBptxAND + hltPreHIL2DoubleMu0 + hltHIDoubleMuLevel1PathL1OpenFiltered + HLTL2muonrecoSequence + hltHIL2DoubleMu0L2Filtered + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL2DoubleMu0_NHitQ_v2 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpenBptxAND + hltPreHIL2DoubleMu0NHitQ + hltHIDoubleMuLevel1PathL1OpenFiltered + HLTL2muonrecoSequence + hltHIL2DoubleMu0L2N1HitsFiltered + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v2 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpenBptxAND + hltPreHIL2DoubleMu0L1HighQL2NHitQ + hltHIDoubleMuLevel1PathL1HighQFiltered + HLTL2muonrecoSequence + hltHIL2DoubleMu0L2N1HitsFiltered + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL2DoubleMu3_v2 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpenBptxAND + hltPreHIL2DoubleMu3 + hltHIDoubleMuLevel1PathL1OpenFiltered + HLTL2muonrecoSequence + hltHIL2DoubleMu3L2Filtered + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL3Mu3_v2 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu3BptxAND + hltPreHIL3Mu3 + hltHIL1SingleMu3withBptxANDL1Filtered + HLTL2muonrecoSequence + hltHIL2Mu3L2Filtered + HLTHIL3muonrecoSequence + hltHISingleMu3L3Filtered + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL3DoubleMuOpen_v2 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpenBptxAND + hltPreHIL1DoubleMuOpenL3 + hltHIDoubleMuLevel1PathL1OpenFiltered + HLTL2muonrecoSequence + hltHIDimuonL2PreFiltered0 + HLTHIL3muonrecoSequence + hltHIDimuonL3FilteredOpen + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL3DoubleMuOpen_Mgt2_v2 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpenBptxAND + hltPreHIL1DoubleMuOpenL3Mgt2 + hltHIDoubleMuLevel1PathL1OpenFiltered + HLTL2muonrecoSequence + hltHIDimuonL2PreFiltered0 + HLTHIL3muonrecoSequence + hltHIDimuonL3FilteredMgt2 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL3DoubleMuOpen_Mgt2_SS_v2 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpenBptxAND + hltPreHIL1DoubleMuOpenL3Mgt2SS + hltHIDoubleMuLevel1PathL1OpenFiltered + HLTL2muonrecoSequence + hltHIDimuonL2PreFiltered0 + HLTHIL3muonrecoSequence + hltHIDimuonL3FilteredMgt2SS + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL3DoubleMuOpen_Mgt2_OS_v2 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpenBptxAND + hltPreHIL1DoubleMuOpenL3Mgt2OS + hltHIDoubleMuLevel1PathL1OpenFiltered + HLTL2muonrecoSequence + hltHIDimuonL2PreFiltered0 + HLTHIL3muonrecoSequence + hltHIDimuonL3FilteredMg2OS + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v2 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpenBptxAND + hltPreHIL1DoubleMuOpenL3Mgt2OSnoCowboy + hltHIDoubleMuLevel1PathL1OpenFiltered + HLTL2muonrecoSequence + hltHIDimuonL2PreFiltered0 + HLTHIL3muonrecoSequence + hltHIDimuonL3FilteredMg2OSnoCowboy + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HISinglePhoton15_v1 = cms.Path( HLTBeginSequence + hltL1sL1SingleEG5BptxAND + hltPreHIPhoton15 + HLTDoCaloSequence + HLTDoHIEcalClusWithCleaningSequence + hltHIPhoton15 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HISinglePhoton20_v2 = cms.Path( HLTBeginSequence + hltL1sL1SingleEG5BptxAND + hltPreHIPhoton20 + HLTDoCaloSequence + HLTDoHIEcalClusWithCleaningSequence + hltHIPhoton20 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HISinglePhoton30_v2 = cms.Path( HLTBeginSequence + hltL1sL1SingleEG5BptxAND + hltPreHIPhoton30 + HLTDoCaloSequence + HLTDoHIEcalClusWithCleaningSequence + hltHIPhoton30 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HISinglePhoton40_v2 = cms.Path( HLTBeginSequence + hltL1sL1SingleEG5BptxAND + hltPreHIPhoton40 + HLTDoCaloSequence + HLTDoHIEcalClusWithCleaningSequence + hltHIPhoton40 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIPhoton10_Photon15_v1 = cms.Path( HLTBeginSequence + hltL1sL1DoubleEG5BptxAND + hltPreHIDoublePhoton10and15 + HLTDoCaloSequence + HLTDoHIEcalClusWithCleaningSequence + hltHIDoublePhoton1015Filter1 + hltHIDoublePhoton1015Filter2 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIPhoton15_Photon20_v1 = cms.Path( HLTBeginSequence + hltL1sL1DoubleEG5BptxAND + hltPreHIDoublePhoton15and20 + HLTDoCaloSequence + HLTDoHIEcalClusWithCleaningSequence + hltHIDoublePhoton1520Filter1 + hltHIDoublePhoton1520Filter2 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIDoublePhoton10_v1 = cms.Path( HLTBeginSequence + hltL1sL1DoubleEG5BptxAND + hltPreHIDoublePhoton10 + HLTDoCaloSequence + HLTDoHIEcalClusWithCleaningSequence + hltHIDoublePhoton10 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIDoublePhoton15_v1 = cms.Path( HLTBeginSequence + hltL1sL1DoubleEG5BptxAND + hltPreHIDoublePhoton15 + HLTDoCaloSequence + HLTDoHIEcalClusWithCleaningSequence + hltHIDoublePhoton15 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIDoublePhoton20_v1 = cms.Path( HLTBeginSequence + hltL1sL1DoubleEG5BptxAND + hltPreHIDoublePhoton20 + HLTDoCaloSequence + HLTDoHIEcalClusWithCleaningSequence + hltHIDoublePhoton20 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIJet55_v1 = cms.Path( HLTBeginSequence + hltL1sL1SingleJet36BptxAND + hltPreHIJet55 + HLTHIRecoJetSequenceIC5Corrected + hltHI1jet55 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIJet65_v1 = cms.Path( HLTBeginSequence + hltL1sL1SingleJet36BptxAND + hltPreHIJet65 + HLTHIRecoJetSequenceIC5Corrected + hltHI1jet65 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIJet80_v1 = cms.Path( HLTBeginSequence + hltL1sL1SingleJet52BptxAND + hltPreHIJet80 + HLTHIRecoJetSequenceIC5Corrected + hltHI1jet80 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIJet95_v1 = cms.Path( HLTBeginSequence + hltL1sL1SingleJet68BptxAND + hltPreHIJet95 + HLTHIRecoJetSequenceIC5Corrected + hltHI1jet95 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIDiJet55_v1 = cms.Path( HLTBeginSequence + hltL1sL1SingleJet36BptxAND + hltPreHIDiJet55and55 + HLTHIRecoJetSequenceIC5Corrected + hltHIDiJet55and55 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIJet65_Jet55_v1 = cms.Path( HLTBeginSequence + hltL1sL1SingleJet36BptxAND + hltPreHIDiJet65and55 + HLTHIRecoJetSequenceIC5Corrected + hltHIDiJet55and55 + hltHI1jet65 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIJetE30_NoBPTX_v1 = cms.Path( HLTBeginSequence + hltL1sL1SingleJet20CentralNotBptx + hltPreJetE30NoBPTX + HLTStoppedHSCPLocalHcalReco + HLTStoppedHSCPJetSequence + hltStoppedHSCPControl1CaloJetEnergy30 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIJetE50_NoBPTX3BX_NoHalo_v1 = cms.Path( HLTBeginSequenceAntiBPTX + hltL1sL1SingleJet32CentralNotBptx + hltL1BeamHaloAntiCoincidence3BX + hltPreJetE50NoBPTX3BXNoHalo + HLTStoppedHSCPLocalHcalReco + hltStoppedHSCPHpdFilter + HLTStoppedHSCPJetSequence + hltStoppedHSCPCaloJetEnergy50 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIActivityHF_Coincidence3_v1 = cms.Path( HLTBeginSequence + hltL1sL1GlobalDecision + hltPreHIActivityHFCoincidence3 + hltHcalDigis + hltHfreco + hltHcalSimpleRecHitFilterCoincidence + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIActivityHF_Single3_v1 = cms.Path( HLTBeginSequence + hltL1sL1GlobalDecision + hltPreHIActivityHFSingle3 + hltHcalDigis + hltHfreco + hltHcalSimpleRecHitFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIClusterVertexCompatibility_v1 = cms.Path( HLTBeginSequence + hltL1sL1GlobalDecision + hltPreHIClusterVertexCompatibility + HLTDoHILocalPixelSequence + hltHIPixelClusterShapeFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HICentralityVeto_v1 = cms.Path( HLTBeginSequence + hltL1sHIMinBiasHfOrBSC + hltPreHICentralityVeto + HLTDoHILocalPixelSequence + hltPixelActivityFilterCentralityVeto + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIFullTrack12_L1Central_v1 = cms.Path( HLTBeginSequenceBPTX + hltL1sETT100BptxAND + hltPreHIFullTrack12L1Central + HLTDoCaloSequence + hltHICaloTowerFilter4 + HLTDoHILocalPixelSequence + HLTPixelSeedingForHITrackTrigger + hltHISinglePixelTrackFilter10 + HLTDoHILocalStripSequenceNonRegional + HLTFullTrackingForHITrackTrigger + hltHISingleFullTrackFilter12 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIFullTrack12_L1Peripheral_v1 = cms.Path( HLTBeginSequenceBPTX + hltL1sJet16CentralNotETT140BptxAND + hltPreHIFullTrack12L1Peripheral + HLTDoCaloSequence + hltHICaloTowerFilter4 + HLTDoHILocalPixelSequence + HLTPixelSeedingForHITrackTrigger + hltHISinglePixelTrackFilter10 + HLTDoHILocalStripSequenceNonRegional + HLTFullTrackingForHITrackTrigger + hltHISingleFullTrackFilter12 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIFullTrack14_L1Central_v1 = cms.Path( HLTBeginSequenceBPTX + hltL1sETT100BptxAND + hltPreHIFullTrack14L1Central + HLTDoCaloSequence + hltHICaloTowerFilter4 + HLTDoHILocalPixelSequence + HLTPixelSeedingForHITrackTrigger + hltHISinglePixelTrackFilter10 + HLTDoHILocalStripSequenceNonRegional + HLTFullTrackingForHITrackTrigger + hltHISingleFullTrackFilter14 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIFullTrack14_L1Peripheral_v1 = cms.Path( HLTBeginSequenceBPTX + hltL1sJet16CentralNotETT140BptxAND + hltPreHIFullTrack14L1Peripheral + HLTDoCaloSequence + hltHICaloTowerFilter4 + HLTDoHILocalPixelSequence + HLTPixelSeedingForHITrackTrigger + hltHISinglePixelTrackFilter10 + HLTDoHILocalStripSequenceNonRegional + HLTFullTrackingForHITrackTrigger + hltHISingleFullTrackFilter14 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIFullTrack20_L1Central_v1 = cms.Path( HLTBeginSequenceBPTX + hltL1sETT100BptxAND + hltPreHIFullTrack20L1Central + HLTDoCaloSequence + hltHICaloTowerFilter4 + HLTDoHILocalPixelSequence + HLTPixelSeedingForHITrackTrigger + hltHISinglePixelTrackFilter10 + HLTDoHILocalStripSequenceNonRegional + HLTFullTrackingForHITrackTrigger + hltHISingleFullTrackFilter20 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIFullTrack20_L1Peripheral_v1 = cms.Path( HLTBeginSequenceBPTX + hltL1sJet16CentralNotETT140BptxAND + hltPreHIFullTrack20L1Peripheral + HLTDoCaloSequence + hltHICaloTowerFilter4 + HLTDoHILocalPixelSequence + HLTPixelSeedingForHITrackTrigger + hltHISinglePixelTrackFilter10 + HLTDoHILocalStripSequenceNonRegional + HLTFullTrackingForHITrackTrigger + hltHISingleFullTrackFilter20 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIFullTrack25_L1Central_v1 = cms.Path( HLTBeginSequenceBPTX + hltL1sETT100BptxAND + hltPreHIFullTrack25L1Central + HLTDoCaloSequence + hltHICaloTowerFilter4 + HLTDoHILocalPixelSequence + HLTPixelSeedingForHITrackTrigger + hltHISinglePixelTrackFilter10 + HLTDoHILocalStripSequenceNonRegional + HLTFullTrackingForHITrackTrigger + hltHISingleFullTrackFilter25 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIFullTrack25_L1Peripheral_v1 = cms.Path( HLTBeginSequenceBPTX + hltL1sJet16CentralNotETT140BptxAND + hltPreHIFullTrack25L1Peripheral + HLTDoCaloSequence + hltHICaloTowerFilter4 + HLTDoHILocalPixelSequence + HLTPixelSeedingForHITrackTrigger + hltHISinglePixelTrackFilter10 + HLTDoHILocalStripSequenceNonRegional + HLTFullTrackingForHITrackTrigger + hltHISingleFullTrackFilter25 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIRandom_v1 = cms.Path( hltRandomEventsFilter + HLTL1UnpackerSequence + hltPreHIRandom + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIUCC010_v2 = cms.Path( HLTBeginSequenceBPTX + hltL1sETT2000 + hltPreHIUCC010 + HLTRecoMETHfSequence + hltGlobalSumETHfFilter3200 + HLTDoHILocalPixelClustersSequence + hltPixelActivityFilter51500 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIUCC015_v2 = cms.Path( HLTBeginSequenceBPTX + hltL1sETT2000 + hltPreHIUCC015 + HLTRecoMETHfSequence + hltGlobalSumETHfFilter3100 + HLTDoHILocalPixelClustersSequence + hltPixelActivityFilter51400 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HICentral10_v2 = cms.Path( HLTBeginSequenceBPTX + hltL1sETT2000 + hltPreHICentral10 + HLTRecoMETHfSequence + hltGlobalSumETHfFilter0 + HLTDoHILocalPixelClustersSequence + hltPixelActivityFilter0 + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIUPCNeuMuPixel_SingleTrack_v1 = cms.Path( HLTBeginSequence + hltL1sHIUPCNeuMu + hltPreHIUPCNeuMuPixelSingleTrack + HLTDoHILocalPixelSequence + HLTPixelTrackingForHITrackTrigger + hltHISinglePixelTrackFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIUPCNeuEG2Pixel_SingleTrack_v1 = cms.Path( HLTBeginSequence + hltL1sHIUPCNeuEG2 + hltPreHIUPCNeuEG2PixelSingleTrack + HLTDoHILocalPixelSequence + HLTPixelTrackingForHITrackTrigger + hltHISinglePixelTrackFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIUPCNeuEG5Pixel_SingleTrack_v1 = cms.Path( HLTBeginSequence + hltL1sHIUPCNeuEG5 + hltPreHIUPCNeuEG5PixelSingleTrack + HLTDoHILocalPixelSequence + HLTPixelTrackingForHITrackTrigger + hltHISinglePixelTrackFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v1 = cms.Path( HLTBeginSequence + hltL1sHIUPCNeuHcalHfMu + hltPreHIUPCNeuHcalHfMuPixelSingleTrack + HLTDoHILocalPixelSequence + HLTPixelTrackingForHITrackTrigger + hltHISinglePixelTrackFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v1 = cms.Path( HLTBeginSequence + hltL1sHIUPCNeuHcalHfEG2 + hltPreHIUPCNeuHcalHfEG2PixelSingleTrack + HLTDoHILocalPixelSequence + HLTPixelTrackingForHITrackTrigger + hltHISinglePixelTrackFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v1 = cms.Path( HLTBeginSequence + hltL1sHIUPCNeuHcalHfEG5 + hltPreHIUPCNeuHcalHfEG5PixelSingleTrack + HLTDoHILocalPixelSequence + HLTPixelTrackingForHITrackTrigger + hltHISinglePixelTrackFilter + HLTDoHIStripZeroSuppression + HLTEndSequence )
HLTriggerFinalPath = cms.Path( hltGtDigis + hltScalersRawToDigi + hltFEDSelector + hltTriggerSummaryAOD + hltTriggerSummaryRAW )
HLTAnalyzerEndpath = cms.EndPath( hltL1GtTrigReport + hltTrigReport )


HLTSchedule = cms.Schedule( *(HLTriggerFirstPath, HLT_HIMET120_v1, HLT_HIMET200_v1, HLT_HIMET220_v1, HLT_HIPhysics_v1, HLT_HIDTCalibration_v1, HLT_HIEcalCalibration_v1, HLT_HIHcalCalibration_v1, HLT_HIZeroBias_v1, HLT_HIZeroBiasXOR_v1, HLT_HIZeroBiasPixel_SingleTrack_v1, HLT_HIMinBiasBSC_v1, HLT_HIMinBiasBSC_OR_v1, HLT_HIMinBiasHF_v1, HLT_HIMinBiasHf_OR_v1, HLT_HIMinBiasHfOrBSC_v1, HLT_HIMinBiasPixel_SingleTrack_v1, HLT_HIMinBiasZDC_Calo_v1, HLT_HIMinBiasZDC_Calo_PlusOrMinus_v1, HLT_HIMinBiasZDCPixel_SingleTrack_v1, HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v1, HLT_HIBptxXOR_v1, HLT_HIL1Algo_BptxXOR_BSC_OR_v1, HLT_HIL1DoubleMuOpen_v2, HLT_HIL1DoubleMu0_HighQ_v2, HLT_HIL2Mu3_v2, HLT_HIL2Mu3_NHitQ_v2, HLT_HIL2Mu7_v2, HLT_HIL2Mu15_v2, HLT_HIL2DoubleMu0_v2, HLT_HIL2DoubleMu0_NHitQ_v2, HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v2, HLT_HIL2DoubleMu3_v2, HLT_HIL3Mu3_v2, HLT_HIL3DoubleMuOpen_v2, HLT_HIL3DoubleMuOpen_Mgt2_v2, HLT_HIL3DoubleMuOpen_Mgt2_SS_v2, HLT_HIL3DoubleMuOpen_Mgt2_OS_v2, HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v2, HLT_HISinglePhoton15_v1, HLT_HISinglePhoton20_v2, HLT_HISinglePhoton30_v2, HLT_HISinglePhoton40_v2, HLT_HIPhoton10_Photon15_v1, HLT_HIPhoton15_Photon20_v1, HLT_HIDoublePhoton10_v1, HLT_HIDoublePhoton15_v1, HLT_HIDoublePhoton20_v1, HLT_HIJet55_v1, HLT_HIJet65_v1, HLT_HIJet80_v1, HLT_HIJet95_v1, HLT_HIDiJet55_v1, HLT_HIJet65_Jet55_v1, HLT_HIJetE30_NoBPTX_v1, HLT_HIJetE50_NoBPTX3BX_NoHalo_v1, HLT_HIActivityHF_Coincidence3_v1, HLT_HIActivityHF_Single3_v1, HLT_HIClusterVertexCompatibility_v1, HLT_HICentralityVeto_v1, HLT_HIFullTrack12_L1Central_v1, HLT_HIFullTrack12_L1Peripheral_v1, HLT_HIFullTrack14_L1Central_v1, HLT_HIFullTrack14_L1Peripheral_v1, HLT_HIFullTrack20_L1Central_v1, HLT_HIFullTrack20_L1Peripheral_v1, HLT_HIFullTrack25_L1Central_v1, HLT_HIFullTrack25_L1Peripheral_v1, HLT_HIRandom_v1, HLT_HIUCC010_v2, HLT_HIUCC015_v2, HLT_HICentral10_v2, HLT_HIUPCNeuMuPixel_SingleTrack_v1, HLT_HIUPCNeuEG2Pixel_SingleTrack_v1, HLT_HIUPCNeuEG5Pixel_SingleTrack_v1, HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v1, HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v1, HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v1, HLTriggerFinalPath, HLTAnalyzerEndpath ))

# Disable HF Noise filters in HIon menu
if 'hltHfreco' in locals():
    hltHfreco.setNoiseFlags = cms.bool( False )

# version specific customizations
import os
cmsswVersion = os.environ['CMSSW_VERSION']

# from CMSSW_5_0_0_pre6: ESSource -> ESProducer in JetMETCorrections/Modules
if cmsswVersion > "CMSSW_5_0":
    if 'hltESSAK5CaloL2L3' in locals():
        hltESSAK5CaloL2L3 = cms.ESProducer( "JetCorrectionESChain",
            appendToDataLabel = cms.string( "" ),
            correctors = cms.vstring( 'hltESSL2RelativeCorrectionService',
                'hltESSL3AbsoluteCorrectionService' ),
            label = cms.string( "hltESSAK5CaloL2L3" )
        )
    if 'hltESSAK5CaloL1L2L3' in locals():
        hltESSAK5CaloL1L2L3 = cms.ESProducer( "JetCorrectionESChain",
            appendToDataLabel = cms.string( "" ),
            correctors = cms.vstring( 'hltESSL1FastJetCorrectionService',
                'hltESSL2RelativeCorrectionService',
                'hltESSL3AbsoluteCorrectionService' ),
            label = cms.string( "hltESSAK5CaloL1L2L3" )
        )
    if 'hltESSL1FastJetCorrectionService' in locals():
        hltESSL1FastJetCorrectionService = cms.ESProducer( "L1FastjetCorrectionESProducer",
            appendToDataLabel = cms.string( "" ),
#           era = cms.string( "Jec10V1" ),
            level = cms.string( "L1FastJet" ),
            algorithm = cms.string( "AK5Calo" ),
#           section = cms.string( "" ),
            srcRho = cms.InputTag( 'hltKT6CaloJets','rho' )#,
#           useCondDB = cms.untracked.bool( True )
        )
    if 'hltESSL2RelativeCorrectionService' in locals():
        hltESSL2RelativeCorrectionService = cms.ESProducer( "LXXXCorrectionESProducer",
            appendToDataLabel = cms.string( "" ),
            level = cms.string( "L2Relative" ),
            algorithm = cms.string( "AK5Calo" )#,
#           section = cms.string( "" ),
#           era = cms.string( "" ),
#           useCondDB = cms.untracked.bool( True )
        )
    if 'hltESSL3AbsoluteCorrectionService' in locals():
        hltESSL3AbsoluteCorrectionService = cms.ESProducer( "LXXXCorrectionESProducer",
            appendToDataLabel = cms.string( "" ),
            level = cms.string( "L3Absolute" ),
            algorithm = cms.string( "AK5Calo" )#,
#           section = cms.string( "" )#,
#           era = cms.string( "" ),
#           useCondDB = cms.untracked.bool( True )
        )

