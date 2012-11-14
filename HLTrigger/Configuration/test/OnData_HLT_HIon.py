# /dev/CMSSW_5_2_6/HIon/V64 (CMSSW_5_2_7_HLT3)

import FWCore.ParameterSet.Config as cms

process = cms.Process( "HLT" )

process.HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_5_2_6/HIon/V64')
)

process.streams = cms.PSet( 
  AForHI = cms.vstring( 'HIDiMuon',
    'HIHighPt',
    'HIMinBiasUPC' ),
  Calibration = cms.vstring( 'TestEnablesEcalHcalDT' ),
  EcalCalibration = cms.vstring( 'EcalLaser' ),
  ExpressForHI = cms.vstring( 'HIExpressPhysics' ),
  HLTDQM = cms.vstring( 'OnlineHltMonitorHI' ),
  HLTMON = cms.vstring( 'OfflineMonitorHI' )
)
process.datasets = cms.PSet( 
  EcalLaser = cms.vstring( 'HLT_HIEcalCalibration_v3' ),
  HIDiMuon = cms.vstring( 'HLT_HIL1DoubleMu0_HighQ_v5',
    'HLT_HIL1DoubleMuOpen_v5',
    'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5',
    'HLT_HIL2DoubleMu0_NHitQ_v5',
    'HLT_HIL2DoubleMu0_v5',
    'HLT_HIL2DoubleMu3_v5',
    'HLT_HIL2Mu15_v5',
    'HLT_HIL2Mu3_NHitQ_v5',
    'HLT_HIL2Mu3_v5',
    'HLT_HIL2Mu7_v5',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_SS_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_v7',
    'HLT_HIL3DoubleMuOpen_v7',
    'HLT_HIL3Mu3_v7' ),
  HIExpressPhysics = cms.vstring( 'HLT_HIBptxXOR_v4',
    'HLT_HICentral10_v7',
    'HLT_HICentralityVeto_v5',
    'HLT_HIDiJet55_v7',
    'HLT_HIDoublePhoton10_v5',
    'HLT_HIDoublePhoton15_v5',
    'HLT_HIDoublePhoton20_v5',
    'HLT_HIFullTrack12_L1Central_v6',
    'HLT_HIFullTrack12_L1Peripheral_v6',
    'HLT_HIFullTrack14_L1Central_v6',
    'HLT_HIFullTrack14_L1Peripheral_v6',
    'HLT_HIFullTrack20_L1Central_v6',
    'HLT_HIFullTrack20_L1Peripheral_v6',
    'HLT_HIFullTrack25_L1Central_v6',
    'HLT_HIFullTrack25_L1Peripheral_v6',
    'HLT_HIJet55_v7',
    'HLT_HIJet65_Jet55_v7',
    'HLT_HIJet65_v7',
    'HLT_HIJet80_v7',
    'HLT_HIJet95_v7',
    'HLT_HIJetE30_NoBPTX_v6',
    'HLT_HIJetE50_NoBPTX3BX_NoHalo_v6',
    'HLT_HIL1Algo_BptxXOR_BSC_OR_v4',
    'HLT_HIL1DoubleMu0_HighQ_v5',
    'HLT_HIL1DoubleMuOpen_v5',
    'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5',
    'HLT_HIL2DoubleMu0_NHitQ_v5',
    'HLT_HIL2DoubleMu0_v5',
    'HLT_HIL2DoubleMu3_v5',
    'HLT_HIL2Mu15_v5',
    'HLT_HIL2Mu3_NHitQ_v5',
    'HLT_HIL2Mu3_v5',
    'HLT_HIL2Mu7_v5',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_SS_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_v7',
    'HLT_HIL3DoubleMuOpen_v7',
    'HLT_HIL3Mu3_v7',
    'HLT_HIMET120_v6',
    'HLT_HIMET200_v6',
    'HLT_HIMET220_v6',
    'HLT_HIMinBiasBSC_OR_v4',
    'HLT_HIMinBiasBSC_v4',
    'HLT_HIMinBiasHF_v4',
    'HLT_HIMinBiasHfOrBSC_v4',
    'HLT_HIMinBiasHf_OR_v4',
    'HLT_HIMinBiasPixel_SingleTrack_v5',
    'HLT_HIMinBiasZDCPixel_SingleTrack_v5',
    'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4',
    'HLT_HIMinBiasZDC_Calo_v4',
    'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5',
    'HLT_HIPhoton10_Photon15_v5',
    'HLT_HIPhoton15_Photon20_v5',
    'HLT_HIPhysics_v4',
    'HLT_HIRandom_v3',
    'HLT_HISinglePhoton15_v5',
    'HLT_HISinglePhoton20_v6',
    'HLT_HISinglePhoton30_v6',
    'HLT_HISinglePhoton40_v6',
    'HLT_HIUCC010_v7',
    'HLT_HIUCC015_v7',
    'HLT_HIUPCNeuEG2Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuEG5Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5',
    'HLT_HIUPCNeuMuPixel_SingleTrack_v5',
    'HLT_HIZeroBiasPixel_SingleTrack_v5',
    'HLT_HIZeroBiasXOR_v4',
    'HLT_HIZeroBias_v4' ),
  HIHighPt = cms.vstring( 'HLT_HIDiJet55_v7',
    'HLT_HIDoublePhoton10_v5',
    'HLT_HIDoublePhoton15_v5',
    'HLT_HIDoublePhoton20_v5',
    'HLT_HIFullTrack12_L1Central_v6',
    'HLT_HIFullTrack12_L1Peripheral_v6',
    'HLT_HIFullTrack14_L1Central_v6',
    'HLT_HIFullTrack14_L1Peripheral_v6',
    'HLT_HIFullTrack20_L1Central_v6',
    'HLT_HIFullTrack20_L1Peripheral_v6',
    'HLT_HIFullTrack25_L1Central_v6',
    'HLT_HIFullTrack25_L1Peripheral_v6',
    'HLT_HIJet55_v7',
    'HLT_HIJet65_Jet55_v7',
    'HLT_HIJet65_v7',
    'HLT_HIJet80_v7',
    'HLT_HIJet95_v7',
    'HLT_HIJetE30_NoBPTX_v6',
    'HLT_HIJetE50_NoBPTX3BX_NoHalo_v6',
    'HLT_HIMET120_v6',
    'HLT_HIMET200_v6',
    'HLT_HIMET220_v6',
    'HLT_HIPhoton10_Photon15_v5',
    'HLT_HIPhoton15_Photon20_v5',
    'HLT_HISinglePhoton15_v5',
    'HLT_HISinglePhoton20_v6',
    'HLT_HISinglePhoton30_v6',
    'HLT_HISinglePhoton40_v6' ),
  HIMinBiasUPC = cms.vstring( 'HLT_HIBptxXOR_v4',
    'HLT_HICentral10_v7',
    'HLT_HICentralityVeto_v5',
    'HLT_HIL1Algo_BptxXOR_BSC_OR_v4',
    'HLT_HIMinBiasBSC_OR_v4',
    'HLT_HIMinBiasBSC_v4',
    'HLT_HIMinBiasHF_v4',
    'HLT_HIMinBiasHfOrBSC_v4',
    'HLT_HIMinBiasHf_OR_v4',
    'HLT_HIMinBiasPixel_SingleTrack_v5',
    'HLT_HIMinBiasZDCPixel_SingleTrack_v5',
    'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4',
    'HLT_HIMinBiasZDC_Calo_v4',
    'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5',
    'HLT_HIPhysics_v4',
    'HLT_HIRandom_v3',
    'HLT_HIUCC010_v7',
    'HLT_HIUCC015_v7',
    'HLT_HIUPCNeuEG2Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuEG5Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5',
    'HLT_HIUPCNeuMuPixel_SingleTrack_v5',
    'HLT_HIZeroBiasPixel_SingleTrack_v5',
    'HLT_HIZeroBiasXOR_v4',
    'HLT_HIZeroBias_v4' ),
  OfflineMonitorHI = cms.vstring( 'HLT_HIActivityHF_Coincidence3_v4',
    'HLT_HIActivityHF_Single3_v4',
    'HLT_HIBptxXOR_v4',
    'HLT_HICentral10_v7',
    'HLT_HICentralityVeto_v5',
    'HLT_HIClusterVertexCompatibility_v5',
    'HLT_HIDiJet55_v7',
    'HLT_HIDoublePhoton10_v5',
    'HLT_HIDoublePhoton15_v5',
    'HLT_HIDoublePhoton20_v5',
    'HLT_HIFullTrack12_L1Central_v6',
    'HLT_HIFullTrack12_L1Peripheral_v6',
    'HLT_HIFullTrack14_L1Central_v6',
    'HLT_HIFullTrack14_L1Peripheral_v6',
    'HLT_HIFullTrack20_L1Central_v6',
    'HLT_HIFullTrack20_L1Peripheral_v6',
    'HLT_HIFullTrack25_L1Central_v6',
    'HLT_HIFullTrack25_L1Peripheral_v6',
    'HLT_HIJet55_v7',
    'HLT_HIJet65_Jet55_v7',
    'HLT_HIJet65_v7',
    'HLT_HIJet80_v7',
    'HLT_HIJet95_v7',
    'HLT_HIJetE30_NoBPTX_v6',
    'HLT_HIJetE50_NoBPTX3BX_NoHalo_v6',
    'HLT_HIL1Algo_BptxXOR_BSC_OR_v4',
    'HLT_HIL1DoubleMu0_HighQ_v5',
    'HLT_HIL1DoubleMuOpen_v5',
    'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5',
    'HLT_HIL2DoubleMu0_NHitQ_v5',
    'HLT_HIL2DoubleMu0_v5',
    'HLT_HIL2DoubleMu3_v5',
    'HLT_HIL2Mu15_v5',
    'HLT_HIL2Mu3_NHitQ_v5',
    'HLT_HIL2Mu3_v5',
    'HLT_HIL2Mu7_v5',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_SS_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_v7',
    'HLT_HIL3DoubleMuOpen_v7',
    'HLT_HIL3Mu3_v7',
    'HLT_HIMET120_v6',
    'HLT_HIMET200_v6',
    'HLT_HIMET220_v6',
    'HLT_HIMinBiasBSC_OR_v4',
    'HLT_HIMinBiasBSC_v4',
    'HLT_HIMinBiasHF_v4',
    'HLT_HIMinBiasHfOrBSC_v4',
    'HLT_HIMinBiasHf_OR_v4',
    'HLT_HIMinBiasPixel_SingleTrack_v5',
    'HLT_HIMinBiasZDCPixel_SingleTrack_v5',
    'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4',
    'HLT_HIMinBiasZDC_Calo_v4',
    'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5',
    'HLT_HIPhoton10_Photon15_v5',
    'HLT_HIPhoton15_Photon20_v5',
    'HLT_HIPhysics_v4',
    'HLT_HIRandom_v3',
    'HLT_HISinglePhoton15_v5',
    'HLT_HISinglePhoton20_v6',
    'HLT_HISinglePhoton30_v6',
    'HLT_HISinglePhoton40_v6',
    'HLT_HIUCC010_v7',
    'HLT_HIUCC015_v7',
    'HLT_HIUPCNeuEG2Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuEG5Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5',
    'HLT_HIUPCNeuMuPixel_SingleTrack_v5',
    'HLT_HIZeroBiasPixel_SingleTrack_v5',
    'HLT_HIZeroBiasXOR_v4',
    'HLT_HIZeroBias_v4' ),
  OnlineHltMonitorHI = cms.vstring( 'HLT_HIActivityHF_Coincidence3_v4',
    'HLT_HIActivityHF_Single3_v4',
    'HLT_HIBptxXOR_v4',
    'HLT_HICentral10_v7',
    'HLT_HICentralityVeto_v5',
    'HLT_HIClusterVertexCompatibility_v5',
    'HLT_HIDiJet55_v7',
    'HLT_HIDoublePhoton10_v5',
    'HLT_HIDoublePhoton15_v5',
    'HLT_HIDoublePhoton20_v5',
    'HLT_HIFullTrack12_L1Central_v6',
    'HLT_HIFullTrack12_L1Peripheral_v6',
    'HLT_HIFullTrack14_L1Central_v6',
    'HLT_HIFullTrack14_L1Peripheral_v6',
    'HLT_HIFullTrack20_L1Central_v6',
    'HLT_HIFullTrack20_L1Peripheral_v6',
    'HLT_HIFullTrack25_L1Central_v6',
    'HLT_HIFullTrack25_L1Peripheral_v6',
    'HLT_HIJet55_v7',
    'HLT_HIJet65_Jet55_v7',
    'HLT_HIJet65_v7',
    'HLT_HIJet80_v7',
    'HLT_HIJet95_v7',
    'HLT_HIJetE30_NoBPTX_v6',
    'HLT_HIJetE50_NoBPTX3BX_NoHalo_v6',
    'HLT_HIL1Algo_BptxXOR_BSC_OR_v4',
    'HLT_HIL1DoubleMu0_HighQ_v5',
    'HLT_HIL1DoubleMuOpen_v5',
    'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5',
    'HLT_HIL2DoubleMu0_NHitQ_v5',
    'HLT_HIL2DoubleMu0_v5',
    'HLT_HIL2DoubleMu3_v5',
    'HLT_HIL2Mu15_v5',
    'HLT_HIL2Mu3_NHitQ_v5',
    'HLT_HIL2Mu3_v5',
    'HLT_HIL2Mu7_v5',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_OS_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_SS_v7',
    'HLT_HIL3DoubleMuOpen_Mgt2_v7',
    'HLT_HIL3DoubleMuOpen_v7',
    'HLT_HIL3Mu3_v7',
    'HLT_HIMET120_v6',
    'HLT_HIMET200_v6',
    'HLT_HIMET220_v6',
    'HLT_HIMinBiasBSC_OR_v4',
    'HLT_HIMinBiasBSC_v4',
    'HLT_HIMinBiasHF_v4',
    'HLT_HIMinBiasHfOrBSC_v4',
    'HLT_HIMinBiasHf_OR_v4',
    'HLT_HIMinBiasPixel_SingleTrack_v5',
    'HLT_HIMinBiasZDCPixel_SingleTrack_v5',
    'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4',
    'HLT_HIMinBiasZDC_Calo_v4',
    'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5',
    'HLT_HIPhoton10_Photon15_v5',
    'HLT_HIPhoton15_Photon20_v5',
    'HLT_HIPhysics_v4',
    'HLT_HIRandom_v3',
    'HLT_HISinglePhoton15_v5',
    'HLT_HISinglePhoton20_v6',
    'HLT_HISinglePhoton30_v6',
    'HLT_HISinglePhoton40_v6',
    'HLT_HIUCC010_v7',
    'HLT_HIUCC015_v7',
    'HLT_HIUPCNeuEG2Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuEG5Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5',
    'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5',
    'HLT_HIUPCNeuMuPixel_SingleTrack_v5',
    'HLT_HIZeroBiasPixel_SingleTrack_v5',
    'HLT_HIZeroBiasXOR_v4',
    'HLT_HIZeroBias_v4' ),
  TestEnablesEcalHcalDT = cms.vstring( 'HLT_HIDTCalibration_v3',
    'HLT_HIEcalCalibration_v3',
    'HLT_HIHcalCalibration_v3' )
)

process.GlobalTag = cms.ESSource( "PoolDBESSource",
    globaltag = cms.string( "GR_H_V32::All" ),
    RefreshEachRun = cms.untracked.bool( True ),
    RefreshOpenIOVs = cms.untracked.bool( False ),
    toGet = cms.VPSet( 
    ),
    DBParameters = cms.PSet( 
      authenticationPath = cms.untracked.string( "." ),
      connectionRetrialTimeOut = cms.untracked.int32( 60 ),
      idleConnectionCleanupPeriod = cms.untracked.int32( 10 ),
      messageLevel = cms.untracked.int32( 0 ),
      enablePoolAutomaticCleanUp = cms.untracked.bool( False ),
      enableConnectionSharing = cms.untracked.bool( True ),
      enableReadOnlySessionOnUpdateConnection = cms.untracked.bool( False ),
      connectionTimeOut = cms.untracked.int32( 0 ),
      connectionRetrialPeriod = cms.untracked.int32( 10 )
    ),
    RefreshAlways = cms.untracked.bool( False ),
    connect = cms.string( "frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG" ),
    ReconnectEachRun = cms.untracked.bool( True ),
    BlobStreamerName = cms.untracked.string( "TBufferBlobStreamingService" ),
    timetype = cms.string( "runnumber" )
)
process.HepPDTESSource = cms.ESSource( "HepPDTESSource",
    pdtFileName = cms.FileInPath( "SimGeneral/HepPDTESSource/data/pythiaparticle.tbl" )
)
process.eegeom = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "EcalMappingRcd" ),
    firstValid = cms.vuint32( 1 )
)
process.es_hardcode = cms.ESSource( "HcalHardcodeCalibrations",
    fromDDD = cms.untracked.bool( False ),
    toGet = cms.untracked.vstring( 'GainWidths' )
)
process.hltESSBTagRecord = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "JetTagComputerRecord" ),
    firstValid = cms.vuint32( 1 )
)
process.hltESSEcalSeverityLevel = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "EcalSeverityLevelAlgoRcd" ),
    firstValid = cms.vuint32( 1 )
)
process.hltESSHcalSeverityLevel = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "HcalSeverityLevelComputerRcd" ),
    firstValid = cms.vuint32( 1 )
)
process.magfield = cms.ESSource( "XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring( 'Geometry/CMSCommonData/data/normal/cmsextent.xml',
      'Geometry/CMSCommonData/data/cms.xml',
      'Geometry/CMSCommonData/data/cmsMagneticField.xml',
      'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml',
      'MagneticField/GeomBuilder/data/MagneticFieldParameters_07_2pi.xml' ),
    rootNodeName = cms.string( "cmsMagneticField:MAGF" )
)

process.hltIter4ESPTrajectoryBuilderITReg = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter4ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter4ESPTrajectoryBuilderITReg" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter4ESPMeasurementTrackerReg" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter4ESPMeasurementTrackerReg = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClustersReg" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter4SiStripClustersReg" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter4ESPMeasurementTrackerReg" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter4ClustersRefRemovalReg" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter3ESPTrajectoryBuilderITReg = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter3ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter3ESPTrajectoryBuilderITReg" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter3ESPMeasurementTrackerReg" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter3ESPMeasurementTrackerReg = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClustersReg" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter3SiStripClustersReg" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter3ESPMeasurementTrackerReg" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter3ClustersRefRemovalReg" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter3ESPLayerTripletsReg = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter3ESPLayerTripletsReg" ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
process.hltIter2ESPTrajectoryBuilderITReg = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter2ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter2ESPTrajectoryBuilderITReg" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter2ESPMeasurementTrackerReg" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter2ESPPixelLayerPairsReg = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter2ESPPixelLayerPairsReg" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter2ESPMeasurementTrackerReg = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClustersReg" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter2SiStripClustersReg" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter2ESPMeasurementTrackerReg" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter2ClustersRefRemovalReg" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter1ESPTrajectoryBuilderITReg = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter1ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter1ESPTrajectoryBuilderITReg" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter1ESPMeasurementTrackerReg" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter1ESPPixelLayerTripletsReg = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltIter1ESPPixelLayerTripletsReg" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1ClustersRefRemovalReg" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter1ESPMeasurementTrackerReg = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClustersReg" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter1SiStripClustersReg" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter1ESPMeasurementTrackerReg" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter1ClustersRefRemovalReg" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltESPTrajectoryBuilderITReg = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltESPTrajectoryBuilderITReg" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTrackerReg" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator9" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPMeasurementTrackerReg = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClustersReg" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltSiStripClustersReg" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltESPMeasurementTrackerReg" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.AnyDirectionAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "AnyDirectionAnalyticalPropagator" ),
  PropagationDirection = cms.string( "anyDirection" )
)
process.AutoMagneticFieldESProducer = cms.ESProducer( "AutoMagneticFieldESProducer",
  label = cms.untracked.string( "" ),
  nominalCurrents = cms.untracked.vint32( -1, 0, 9558, 14416, 16819, 18268, 19262 ),
  valueOverride = cms.int32( -1 ),
  mapLabels = cms.untracked.vstring( '090322_3_8t',
    '0t',
    '071212_2t',
    '071212_3t',
    '071212_3_5t',
    '090322_3_8t',
    '071212_4t' )
)
process.CSCGeometryESModule = cms.ESProducer( "CSCGeometryESModule",
  useRealWireGeometry = cms.bool( True ),
  appendToDataLabel = cms.string( "" ),
  alignmentsLabel = cms.string( "" ),
  useGangedStripsInME1a = cms.bool( True ),
  debugV = cms.untracked.bool( False ),
  useOnlyWiresInME1a = cms.bool( False ),
  useDDD = cms.bool( False ),
  useCentreTIOffsets = cms.bool( False ),
  applyAlignment = cms.bool( True )
)
process.CaloGeometryBuilder = cms.ESProducer( "CaloGeometryBuilder",
  SelectedCalos = cms.vstring( 'HCAL',
    'ZDC',
    'EcalBarrel',
    'EcalEndcap',
    'EcalPreshower',
    'TOWER' )
)
process.CaloTopologyBuilder = cms.ESProducer( "CaloTopologyBuilder" )
process.CaloTowerConstituentsMapBuilder = cms.ESProducer( "CaloTowerConstituentsMapBuilder",
  MapFile = cms.untracked.string( "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz" )
)
process.CaloTowerGeometryFromDBEP = cms.ESProducer( "CaloTowerGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
process.CastorDbProducer = cms.ESProducer( "CastorDbProducer",
  appendToDataLabel = cms.string( "" )
)
process.ClusterShapeHitFilterESProducer = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  ComponentName = cms.string( "ClusterShapeHitFilter" )
)
process.DTGeometryESModule = cms.ESProducer( "DTGeometryESModule",
  appendToDataLabel = cms.string( "" ),
  fromDDD = cms.bool( False ),
  applyAlignment = cms.bool( True ),
  alignmentsLabel = cms.string( "" )
)
process.EcalBarrelGeometryFromDBEP = cms.ESProducer( "EcalBarrelGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
process.EcalElectronicsMappingBuilder = cms.ESProducer( "EcalElectronicsMappingBuilder" )
process.EcalEndcapGeometryFromDBEP = cms.ESProducer( "EcalEndcapGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
process.EcalLaserCorrectionService = cms.ESProducer( "EcalLaserCorrectionService" )
process.EcalPreshowerGeometryFromDBEP = cms.ESProducer( "EcalPreshowerGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
process.EcalUnpackerWorkerESProducer = cms.ESProducer( "EcalUnpackerWorkerESProducer",
  CalibRHAlgo = cms.PSet( 
    flagsMapDBReco = cms.vint32( 0, 0, 0, 0, 4, -1, -1, -1, 4, 4, 7, 7, 7, 8, 9 ),
    Type = cms.string( "EcalRecHitWorkerSimple" ),
    killDeadChannels = cms.bool( True ),
    ChannelStatusToBeExcluded = cms.vint32( 10, 11, 12, 13, 14 ),
    laserCorrection = cms.bool( True ),
    EBLaserMIN = cms.double( 0.5 ),
    EELaserMIN = cms.double( 0.5 ),
    EBLaserMAX = cms.double( 2.0 ),
    EELaserMAX = cms.double( 3.0 )
  ),
  ComponentName = cms.string( "" ),
  UncalibRHAlgo = cms.PSet(  Type = cms.string( "EcalUncalibRecHitWorkerWeights" ) ),
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
  )
)
process.HcalGeometryFromDBEP = cms.ESProducer( "HcalGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
process.HcalTopologyIdealEP = cms.ESProducer( "HcalTopologyIdealEP" )
process.MaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterial" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.MaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialForHI" ),
  Mass = cms.double( 0.139 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.OppositeMaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialOpposite" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.OppositeMaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  Mass = cms.double( 0.139 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.RPCGeometryESModule = cms.ESProducer( "RPCGeometryESModule",
  useDDD = cms.untracked.bool( False ),
  compatibiltyWith11 = cms.untracked.bool( True )
)
process.SiStripGainESProducer = cms.ESProducer( "SiStripGainESProducer",
  printDebug = cms.untracked.bool( False ),
  appendToDataLabel = cms.string( "" ),
  APVGain = cms.VPSet( 
    cms.PSet(  Record = cms.string( "SiStripApvGainRcd" ),
      NormalizationFactor = cms.untracked.double( 1.0 ),
      Label = cms.untracked.string( "" )
    ),
    cms.PSet(  Record = cms.string( "SiStripApvGain2Rcd" ),
      NormalizationFactor = cms.untracked.double( 1.0 ),
      Label = cms.untracked.string( "" )
    )
  ),
  AutomaticNormalization = cms.bool( False )
)
process.SiStripQualityESProducer = cms.ESProducer( "SiStripQualityESProducer",
  appendToDataLabel = cms.string( "" ),
  PrintDebugOutput = cms.bool( False ),
  PrintDebug = cms.untracked.bool( False ),
  ListOfRecordToMerge = cms.VPSet( 
    cms.PSet(  record = cms.string( "SiStripDetVOffRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripDetCablingRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripBadChannelRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripBadFiberRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripBadModuleRcd" ),
      tag = cms.string( "" )
    )
  ),
  UseEmptyRunInfo = cms.bool( False ),
  ReduceGranularity = cms.bool( False ),
  ThresholdForReducedGranularity = cms.double( 0.3 )
)
process.SiStripRecHitMatcherESProducer = cms.ESProducer( "SiStripRecHitMatcherESProducer",
  ComponentName = cms.string( "StandardMatcher" ),
  NSigmaInside = cms.double( 3.0 )
)
process.SlaveField0 = cms.ESProducer( "UniformMagneticFieldESProducer",
  ZFieldInTesla = cms.double( 0.0 ),
  label = cms.untracked.string( "slave_0" )
)
process.SlaveField20 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "2_0T" ) ),
  label = cms.untracked.string( "slave_20" )
)
process.SlaveField30 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "3_0T" ) ),
  label = cms.untracked.string( "slave_30" )
)
process.SlaveField35 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "3_5T" ) ),
  label = cms.untracked.string( "slave_35" )
)
process.SlaveField38 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "3_8T" ) ),
  label = cms.untracked.string( "slave_38" )
)
process.SlaveField40 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  version = cms.string( "OAE_1103l_071212" ),
  parameters = cms.PSet(  BValue = cms.string( "4_0T" ) ),
  label = cms.untracked.string( "slave_40" )
)
process.SteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "anyDirection" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "SteppingHelixPropagatorAny" )
)
process.StripCPEfromTrackAngleESProducer = cms.ESProducer( "StripCPEESProducer",
  TanDiffusionAngle = cms.double( 0.01 ),
  UncertaintyScaling = cms.double( 1.42 ),
  ThicknessRelativeUncertainty = cms.double( 0.02 ),
  MaybeNoiseThreshold = cms.double( 3.5 ),
  ComponentName = cms.string( "StripCPEfromTrackAngle" ),
  MinimumUncertainty = cms.double( 0.01 ),
  NoiseThreshold = cms.double( 2.3 )
)
process.TrackerDigiGeometryESModule = cms.ESProducer( "TrackerDigiGeometryESModule",
  appendToDataLabel = cms.string( "" ),
  fromDDD = cms.bool( False ),
  applyAlignment = cms.bool( True ),
  alignmentsLabel = cms.string( "" )
)
process.TrackerGeometricDetESModule = cms.ESProducer( "TrackerGeometricDetESModule",
  fromDDD = cms.bool( False )
)
process.TransientTrackBuilderESProducer = cms.ESProducer( "TransientTrackBuilderESProducer",
  ComponentName = cms.string( "TransientTrackBuilder" )
)
process.VBF0 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32(  ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble(  ),
  label = cms.untracked.string( "0t" ),
  version = cms.string( "grid_1103l_071212_2t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_0" ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.VBF20 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32(  ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble(  ),
  label = cms.untracked.string( "071212_2t" ),
  version = cms.string( "grid_1103l_071212_2t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_20" ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.VBF30 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32(  ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble(  ),
  label = cms.untracked.string( "071212_3t" ),
  version = cms.string( "grid_1103l_071212_3t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_30" ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.VBF35 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32(  ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble(  ),
  label = cms.untracked.string( "071212_3_5t" ),
  version = cms.string( "grid_1103l_071212_3_5t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_35" ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.VBF38 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32( 14100, 14200, 17600, 17800, 17900, 18100, 18300, 18400, 18600, 23100, 23300, 23400, 23600, 23800, 23900, 24100, 28600, 28800, 28900, 29100, 29300, 29400, 29600, 28609, 28809, 28909, 29109, 29309, 29409, 29609, 28610, 28810, 28910, 29110, 29310, 29410, 29610, 28611, 28811, 28911, 29111, 29311, 29411, 29611 ),
  overrideMasterSector = cms.bool( False ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble( 1.0, 1.0, 0.994, 1.004, 1.004, 1.005, 1.004, 1.004, 0.994, 0.965, 0.958, 0.958, 0.953, 0.958, 0.958, 0.965, 0.918, 0.924, 0.924, 0.906, 0.924, 0.924, 0.918, 0.991, 0.998, 0.998, 0.978, 0.998, 0.998, 0.991, 0.991, 0.998, 0.998, 0.978, 0.998, 0.998, 0.991, 0.991, 0.998, 0.998, 0.978, 0.998, 0.998, 0.991 ),
  label = cms.untracked.string( "090322_3_8t" ),
  version = cms.string( "grid_1103l_090322_3_8t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_38" ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.VBF40 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  scalingVolumes = cms.vint32(  ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  scalingFactors = cms.vdouble(  ),
  label = cms.untracked.string( "071212_4t" ),
  version = cms.string( "grid_1103l_071212_4t" ),
  debugBuilder = cms.untracked.bool( False ),
  paramLabel = cms.string( "slave_40" ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.ZdcGeometryFromDBEP = cms.ESProducer( "ZdcGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
process.caloDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "CaloDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.cosmicsNavigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "CosmicNavigationSchool" )
)
process.ecalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "EcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.02 ),
  nEta = cms.int32( 300 ),
  nPhi = cms.int32( 360 ),
  includeBadChambers = cms.bool( False )
)
process.ecalSeverityLevel = cms.ESProducer( "EcalSeverityLevelESProducer",
  dbstatusMask = cms.PSet( 
    kGood = cms.vuint32( 0 ),
    kProblematic = cms.vuint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ),
    kRecovered = cms.vuint32(  ),
    kTime = cms.vuint32(  ),
    kWeird = cms.vuint32(  ),
    kBad = cms.vuint32( 11, 12, 13, 14, 15, 16 )
  ),
  timeThresh = cms.double( 2.0 ),
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
  )
)
process.hcalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.hcalRecAlgos = cms.ESProducer( "HcalRecAlgoESProducer",
  RecoveredRecHitBits = cms.vstring( 'TimingAddedBit',
    'TimingSubtractedBit' ),
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
  'TimingErrorBit',
  'HBHEFlatNoise',
  'HBHESpikeNoise',
  'HBHETriangleNoise',
  'HBHETS4TS5Noise' ),
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
  DropChannelStatusBits = cms.vstring( 'HcalCellMask',
    'HcalCellOff',
    'HcalCellDead' )
)
process.hcal_db_producer = cms.ESProducer( "HcalDbProducer" )
process.hltCombinedSecondaryVertex = cms.ESProducer( "CombinedSecondaryVertexESProducer",
  categoryVariableName = cms.string( "vertexCategory" ),
  useTrackWeights = cms.bool( True ),
  useCategories = cms.bool( True ),
  pseudoMultiplicityMin = cms.uint32( 2 ),
  correctVertexMass = cms.bool( True ),
  trackSelection = cms.PSet( 
    totalHitsMin = cms.uint32( 0 ),
    jetDeltaRMax = cms.double( 0.3 ),
    qualityClass = cms.string( "any" ),
    pixelHitsMin = cms.uint32( 0 ),
    sip3dSigMin = cms.double( -99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    normChi2Max = cms.double( 99999.9 ),
    maxDistToAxis = cms.double( 0.07 ),
    sip2dValMax = cms.double( 99999.9 ),
    maxDecayLen = cms.double( 5.0 ),
    ptMin = cms.double( 0.0 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip2dSigMin = cms.double( -99999.9 ),
    sip3dValMax = cms.double( 99999.9 ),
    sip2dValMin = cms.double( -99999.9 ),
    sip3dValMin = cms.double( -99999.9 )
  ),
  calibrationRecords = cms.vstring( 'CombinedSVRecoVertex',
    'CombinedSVPseudoVertex',
    'CombinedSVNoVertex' ),
  trackPairV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.03 ) ),
  charmCut = cms.double( 1.5 ),
  vertexFlip = cms.bool( False ),
  minimumTrackWeight = cms.double( 0.5 ),
  pseudoVertexV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.05 ) ),
  trackMultiplicityMin = cms.uint32( 3 ),
  trackPseudoSelection = cms.PSet( 
    totalHitsMin = cms.uint32( 0 ),
    jetDeltaRMax = cms.double( 0.3 ),
    qualityClass = cms.string( "any" ),
    pixelHitsMin = cms.uint32( 0 ),
    sip3dSigMin = cms.double( -99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    normChi2Max = cms.double( 99999.9 ),
    maxDistToAxis = cms.double( 0.07 ),
    sip2dValMax = cms.double( 99999.9 ),
    maxDecayLen = cms.double( 5.0 ),
    ptMin = cms.double( 0.0 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip2dSigMin = cms.double( 2.0 ),
    sip3dValMax = cms.double( 99999.9 ),
    sip2dValMin = cms.double( -99999.9 ),
    sip3dValMin = cms.double( -99999.9 )
  ),
  trackSort = cms.string( "sip2dSig" ),
  trackFlip = cms.bool( False )
)
process.hltESPAK5CaloL1L2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL1FastJetCorrectionESProducer',
    'hltESPL2RelativeCorrectionESProducer',
    'hltESPL3AbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPAK5CaloL2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL2RelativeCorrectionESProducer',
    'hltESPL3AbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPAK5PFL1L2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL1PFFastJetCorrectionESProducer',
    'hltESPL2PFRelativeCorrectionESProducer',
    'hltESPL3PFAbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPAK5PFNoPUL1L2L3 = cms.ESProducer( "JetCorrectionESChain",
  correctors = cms.vstring( 'hltESPL1PFNoPUFastJetCorrectionESProducer',
    'hltESPL2PFNoPURelativeCorrectionESProducer',
    'hltESPL3PFNoPUAbsoluteCorrectionESProducer' ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPAnalyticalPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" )
)
process.hltESPBwdAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPBwdAnalyticalPropagator" ),
  PropagationDirection = cms.string( "oppositeToMomentum" )
)
process.hltESPBwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPBwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.hltESPChi2EstimatorForRefit = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 100000.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2EstimatorForRefit" )
)
process.hltESPChi2MeasurementEstimator = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator" )
)
process.hltESPChi2MeasurementEstimator16 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator16" )
)
process.hltESPChi2MeasurementEstimator9 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator9" )
)
process.hltESPCkf3HitTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPCkf3HitTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPCkf3HitTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPCkf3HitTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltESPCkf3HitTrajectoryFilter" )
)
process.hltESPCkfTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPCkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPCkfTrajectoryBuilderForHI = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterialForHI" ),
  trajectoryFilterName = cms.string( "hltESPCkfTrajectoryFilterForHI" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPCkfTrajectoryBuilderForHI" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTrackerForHI" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPCkfTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltESPCkfTrajectoryFilter" )
)
process.hltESPCkfTrajectoryFilterForHI = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltESPCkfTrajectoryFilterForHI" )
)
process.hltESPCloseComponentsMerger5D = cms.ESProducer( "CloseComponentsMergerESProducer5D",
  ComponentName = cms.string( "hltESPCloseComponentsMerger5D" ),
  MaxComponents = cms.int32( 12 ),
  DistanceMeasure = cms.string( "hltESPKullbackLeiblerDistance5D" )
)
process.hltESPDummyDetLayerGeometry = cms.ESProducer( "DetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPESUnpackerWorker = cms.ESProducer( "ESUnpackerWorkerESProducer",
  RHAlgo = cms.PSet( 
    ESRecoAlgo = cms.int32( 0 ),
    Type = cms.string( "ESRecHitWorker" )
  ),
  DCCDataUnpacker = cms.PSet(  LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ) ),
  ComponentName = cms.string( "hltESPESUnpackerWorker" )
)
process.hltESPEcalRegionCablingESProducer = cms.ESProducer( "EcalRegionCablingESProducer",
  esMapping = cms.PSet(  LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ) )
)
process.hltESPEcalTrigTowerConstituentsMapBuilder = cms.ESProducer( "EcalTrigTowerConstituentsMapBuilder",
  MapFile = cms.untracked.string( "Geometry/EcalMapping/data/EndCap_TTMap.txt" )
)
process.hltESPElectronChi2 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 2000.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPElectronChi2" )
)
process.hltESPElectronMaterialEffects = cms.ESProducer( "GsfMaterialEffectsESProducer",
  BetheHeitlerParametrization = cms.string( "BetheHeitler_cdfmom_nC6_O5.par" ),
  EnergyLossUpdator = cms.string( "GsfBetheHeitlerUpdator" ),
  ComponentName = cms.string( "hltESPElectronMaterialEffects" ),
  MultipleScatteringUpdator = cms.string( "MultipleScatteringUpdator" ),
  Mass = cms.double( 5.11E-4 ),
  BetheHeitlerCorrection = cms.int32( 2 )
)
process.hltESPFastSteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "anyDirection" ),
  useTuningForL2Speed = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPFastSteppingHelixPropagatorAny" )
)
process.hltESPFastSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useTuningForL2Speed = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" )
)
process.hltESPFittingSmootherIT = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( 10.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  MinNumberOfHits = cms.int32( 3 ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  ComponentName = cms.string( "hltESPFittingSmootherIT" ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  RejectTracks = cms.bool( True )
)
process.hltESPFittingSmootherRK = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  ComponentName = cms.string( "hltESPFittingSmootherRK" ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  RejectTracks = cms.bool( True )
)
process.hltESPFwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPFwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.hltESPGlobalDetLayerGeometry = cms.ESProducer( "GlobalDetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPGlobalDetLayerGeometry" )
)
process.hltESPGlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" )
process.hltESPGsfElectronFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPGsfTrajectoryFitter" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPGsfTrajectorySmoother" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  ComponentName = cms.string( "hltESPGsfElectronFittingSmoother" ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  RejectTracks = cms.bool( True )
)
process.hltESPGsfTrajectoryFitter = cms.ESProducer( "GsfTrajectoryFitterESProducer",
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  ComponentName = cms.string( "hltESPGsfTrajectoryFitter" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  GeometricalPropagator = cms.string( "hltESPAnalyticalPropagator" )
)
process.hltESPGsfTrajectorySmoother = cms.ESProducer( "GsfTrajectorySmootherESProducer",
  ErrorRescaling = cms.double( 100.0 ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  ComponentName = cms.string( "hltESPGsfTrajectorySmoother" ),
  GeometricalPropagator = cms.string( "hltESPBwdAnalyticalPropagator" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" )
)
process.hltESPHIMixedLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltESPHIMixedLayerPairs" ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  FPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0036 ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    useErrorsFromParam = cms.bool( True )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0060 ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    useErrorsFromParam = cms.bool( True )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltESPHIPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltESPHIPixelLayerPairs" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltESPHIPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltESPHIPixelLayerTriplets" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltESPHITTRHBuilderWithoutRefit = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "Fake" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "Fake" ),
  ComponentName = cms.string( "hltESPHITTRHBuilderWithoutRefit" )
)
process.hltESPKFFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPKFTrajectoryFitter" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPKFTrajectorySmoother" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  ComponentName = cms.string( "hltESPKFFittingSmoother" ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  RejectTracks = cms.bool( True )
)
process.hltESPKFFittingSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  ComponentName = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  RejectTracks = cms.bool( True )
)
process.hltESPKFFittingSmootherWithOutliersRejectionAndRK = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( 20.0 ),
  LogPixelProbabilityCut = cms.double( -14.0 ),
  Fitter = cms.string( "hltESPRKFitter" ),
  MinNumberOfHits = cms.int32( 3 ),
  Smoother = cms.string( "hltESPRKSmoother" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  ComponentName = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  RejectTracks = cms.bool( True )
)
process.hltESPKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmoother" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectorySmootherForMuonTrackLoader = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFUpdator = cms.ESProducer( "KFUpdatorESProducer",
  ComponentName = cms.string( "hltESPKFUpdator" )
)
process.hltESPKullbackLeiblerDistance5D = cms.ESProducer( "DistanceBetweenComponentsESProducer5D",
  ComponentName = cms.string( "hltESPKullbackLeiblerDistance5D" ),
  DistanceMeasure = cms.string( "KullbackLeibler" )
)
process.hltESPL1FastJetCorrectionESProducer = cms.ESProducer( "L1FastjetCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  srcRho = cms.InputTag( 'hltKT6CaloJets','rho' ),
  algorithm = cms.string( "AK5CaloHLT" ),
  level = cms.string( "L1FastJet" )
)
process.hltESPL1PFFastJetCorrectionESProducer = cms.ESProducer( "L1FastjetCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  srcRho = cms.InputTag( 'hltKT6PFJets','rho' ),
  algorithm = cms.string( "AK5PFHLT" ),
  level = cms.string( "L1FastJet" )
)
process.hltESPL1PFNoPUFastJetCorrectionESProducer = cms.ESProducer( "L1FastjetCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  srcRho = cms.InputTag( 'hltKT6PFJets','rho' ),
  algorithm = cms.string( "AK5PFchsHLT" ),
  level = cms.string( "L1FastJet" )
)
process.hltESPL2PFNoPURelativeCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFchsHLT" ),
  level = cms.string( "L2Relative" )
)
process.hltESPL2PFRelativeCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFHLT" ),
  level = cms.string( "L2Relative" )
)
process.hltESPL2RelativeCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5CaloHLT" ),
  level = cms.string( "L2Relative" )
)
process.hltESPL3AbsoluteCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5CaloHLT" ),
  level = cms.string( "L3Absolute" )
)
process.hltESPL3MuKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPSmartPropagatorAny" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPL3PFAbsoluteCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFHLT" ),
  level = cms.string( "L3Absolute" )
)
process.hltESPL3PFNoPUAbsoluteCorrectionESProducer = cms.ESProducer( "LXXXCorrectionESProducer",
  appendToDataLabel = cms.string( "" ),
  algorithm = cms.string( "AK5PFchsHLT" ),
  level = cms.string( "L3Absolute" )
)
process.hltESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltSiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltESPMeasurementTrackerForHI = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( False ),
  Regional = cms.bool( False ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltHISiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripRawToDigi' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltHISiStripClustersNonRegional" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltESPMeasurementTrackerForHI" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltESPMixedLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltESPMixedLayerPairs" ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltESPMuTrackJpsiEffTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPMuTrackJpsiEffTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltESPMuTrackJpsiEffTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPMuTrackJpsiEffTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    minPt = cms.double( 1.0 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 9 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    nSigmaMinPt = cms.double( 5.0 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltESPMuTrackJpsiEffTrajectoryFilter" )
)
process.hltESPMuTrackJpsiTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPMuTrackJpsiTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltESPMuTrackJpsiTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPMuTrackJpsiTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltESPMuTrackJpsiTrajectoryFilter" )
)
process.hltESPMuonCkfTrajectoryBuilder = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPMuonCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPMuonCkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  useSeedLayer = cms.bool( False ),
  deltaEta = cms.double( 0.1 ),
  deltaPhi = cms.double( 0.1 ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  rescaleErrorIfFail = cms.double( 1.0 ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPMuonCkfTrajectoryBuilderSeedHit = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPMuonCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPMuonCkfTrajectoryBuilderSeedHit" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  useSeedLayer = cms.bool( True ),
  deltaEta = cms.double( 0.1 ),
  deltaPhi = cms.double( 0.1 ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  rescaleErrorIfFail = cms.double( 1.0 ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPMuonCkfTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltESPMuonCkfTrajectoryFilter" )
)
process.hltESPMuonDetLayerGeometryESProducer = cms.ESProducer( "MuonDetLayerGeometryESProducer" )
process.hltESPMuonTransientTrackingRecHitBuilder = cms.ESProducer( "MuonTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
)
process.hltESPPixelCPEGeneric = cms.ESProducer( "PixelCPEGenericESProducer",
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
  TanLorentzAnglePerTesla = cms.double( 0.106 ),
  inflate_errors = cms.bool( False ),
  eff_charge_cut_lowX = cms.double( 0.0 ),
  eff_charge_cut_highY = cms.double( 1.0 ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  EdgeClusterErrorY = cms.double( 85.0 ),
  ComponentName = cms.string( "hltESPPixelCPEGeneric" ),
  eff_charge_cut_lowY = cms.double( 0.0 ),
  PixelErrorParametrization = cms.string( "NOTcmsim" ),
  Alpha2Order = cms.bool( True )
)
process.hltESPPixelCPETemplateReco = cms.ESProducer( "PixelCPETemplateRecoESProducer",
  DoCosmics = cms.bool( False ),
  LoadTemplatesFromDB = cms.bool( True ),
  ComponentName = cms.string( "hltESPPixelCPETemplateReco" ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  speed = cms.int32( -2 ),
  UseClusterSplitter = cms.bool( False )
)
process.hltESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltESPPixelLayerPairs" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltESPPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltESPPixelLayerTriplets" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltESPPixelLayerTripletsHITHB = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3' ),
  ComponentName = cms.string( "hltESPPixelLayerTripletsHITHB" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltESPPixelLayerTripletsHITHE = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltESPPixelLayerTripletsHITHE" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltESPPixelLayerTripletsReg = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltESPPixelLayerTripletsReg" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0036 ),
    hitErrorRPhi = cms.double( 0.0051 ),
    useErrorsFromParam = cms.bool( True ),
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0060 ),
    hitErrorRPhi = cms.double( 0.0027 ),
    useErrorsFromParam = cms.bool( True ),
    HitProducer = cms.string( "hltSiPixelRecHitsReg" ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltESPPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
  maxImpactParameterSig = cms.double( 999999.0 ),
  deltaR = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  impactParameterType = cms.int32( 0 ),
  trackQualityClass = cms.string( "any" ),
  deltaRmin = cms.double( 0.0 ),
  maxImpactParameter = cms.double( 0.03 ),
  maximumDistanceToJetAxis = cms.double( 999999.0 ),
  nthTrack = cms.int32( -1 )
)
process.hltESPRKTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPRKFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
process.hltESPRKTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPRKSmoother" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
process.hltESPRungeKuttaTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( True )
)
process.hltESPSiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 )
)
process.hltESPSmartPropagator = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagator" )
)
process.hltESPSmartPropagatorAny = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorAny" )
)
process.hltESPSmartPropagatorAnyOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorAnyOpposite" )
)
process.hltESPSmartPropagatorOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorOpposite" )
)
process.hltESPSoftLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  distance = cms.double( 0.5 )
)
process.hltESPSoftLeptonByPt = cms.ESProducer( "LeptonTaggerByPtESProducer",
  ipSign = cms.string( "any" )
)
process.hltESPSteppingHelixPropagatorAlong = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPSteppingHelixPropagatorAlong" )
)
process.hltESPSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPSteppingHelixPropagatorOpposite" )
)
process.hltESPStraightLinePropagator = cms.ESProducer( "StraightLinePropagatorESProducer",
  ComponentName = cms.string( "hltESPStraightLinePropagator" ),
  PropagationDirection = cms.string( "alongMomentum" )
)
process.hltESPTTRHBWithTrackAngle = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBWithTrackAngle" )
)
process.hltESPTTRHBuilderAngleAndTemplate = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPETemplateReco" ),
  ComponentName = cms.string( "hltESPTTRHBuilderAngleAndTemplate" )
)
process.hltESPTTRHBuilderPixelOnly = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBuilderPixelOnly" )
)
process.hltESPTTRHBuilderWithoutAngle4PixelTriplets = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
)
process.hltESPTrackCounting3D1st = cms.ESProducer( "TrackCountingESProducer",
  deltaR = cms.double( -1.0 ),
  maximumDistanceToJetAxis = cms.double( 0.07 ),
  impactParameterType = cms.int32( 0 ),
  trackQualityClass = cms.string( "any" ),
  maximumDecayLength = cms.double( 5.0 ),
  nthTrack = cms.int32( 1 )
)
process.hltESPTrackCounting3D2nd = cms.ESProducer( "TrackCountingESProducer",
  deltaR = cms.double( -1.0 ),
  maximumDistanceToJetAxis = cms.double( 0.07 ),
  impactParameterType = cms.int32( 0 ),
  trackQualityClass = cms.string( "any" ),
  maximumDecayLength = cms.double( 5.0 ),
  nthTrack = cms.int32( 2 )
)
process.hltESPTrackerRecoGeometryESProducer = cms.ESProducer( "TrackerRecoGeometryESProducer",
  appendToDataLabel = cms.string( "" ),
  trackerGeometryLabel = cms.untracked.string( "" )
)
process.hltESPTrajectoryBuilderForElectrons = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "hltESPFwdElectronPropagator" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterForElectrons" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPTrajectoryBuilderForElectrons" ),
  propagatorOpposite = cms.string( "hltESPBwdElectronPropagator" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPElectronChi2" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 90.0 )
)
process.hltESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator9" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPTrajectoryBuilderL3 = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterL3" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltESPTrajectoryBuilderL3" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( False )
)
process.hltESPTrajectoryCleanerBySharedSeeds = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedSeeds" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedSeeds" ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPTrajectoryFilterForElectrons = cms.ESProducer( "TrajectoryFilterESProducer",
  filterPset = cms.PSet( 
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    minPt = cms.double( 2.0 ),
    minHitsMinPt = cms.int32( -1 ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minimumNumberOfHits = cms.int32( 5 ),
    chargeSignificance = cms.double( -1.0 )
  ),
  ComponentName = cms.string( "hltESPTrajectoryFilterForElectrons" )
)
process.hltESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltESPTrajectoryFilterIT" )
)
process.hltESPTrajectoryFilterL3 = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltESPTrajectoryFilterL3" )
)
process.hltESPTrajectoryFitterRK = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectoryFitterRK" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPTrajectorySmootherRK = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectorySmootherRK" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPbJetRegionalTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPbJetRegionalTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltESPbJetRegionalTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltESPbJetRegionalTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltESPbJetRegionalTrajectoryFilter" )
)
process.hltHIAllESPCkf3HitTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPCkf3HitTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltHIAllESPCkf3HitTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltHIAllESPCkfTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltHIAllESPCkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltHIAllESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltHISiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltHISiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltHISiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltHIAllESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltHIAllESPMuonCkfTrajectoryBuilder = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPMuonCkfTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltHIAllESPMuonCkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  useSeedLayer = cms.bool( False ),
  deltaEta = cms.double( 0.1 ),
  deltaPhi = cms.double( 0.1 ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  rescaleErrorIfFail = cms.double( 1.0 ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltHIAllESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 5 ),
  ComponentName = cms.string( "hltHIAllESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter1ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter1SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter1ESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter1ESPPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltIter1ESPPixelLayerTriplets" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter1ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter1ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter1ESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter1ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter1ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltIter1ESPTrajectoryFilterIT" )
)
process.hltIter3ESPLayerTripletsPA = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter3ESPLayerTripletsPA" ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltPAFullTrackIter3ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltPAFullTrackIter3ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
process.hltIter3ESPTrajectoryBuilderITPA = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter3ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter3ESPTrajectoryBuilderITPA" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter3ESPMeasurementTrackerPA" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter3ESPMeasurementTrackerPA = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltPAFullTrackIter3SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter3ESPMeasurementTrackerPA" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltPAFullTrackIter3ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter1ESPPixelLayerTripletsPA = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltIter1ESPPixelLayerTripletsPA" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltPAFullTrackIter1ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltPAFullTrackIter1ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter2ESPPixelLayerPairsPA = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter2ESPPixelLayerPairsPA" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltPAFullTrackIter2ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltPAFullTrackIter2ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter2ESPTrajectoryBuilderITPA = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter2ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter2ESPTrajectoryBuilderITPA" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter2ESPMeasurementTrackerPA" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter2ESPMeasurementTrackerPA = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltPAFullTrackIter2SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter2ESPMeasurementTrackerPA" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltPAFullTrackIter2ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter1ESPTrajectoryBuilderITPA = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter1ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter1ESPTrajectoryBuilderITPA" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter1ESPMeasurementTrackerPA" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter1ESPMeasurementTrackerPA = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltPAFullTrackIter1SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter1ESPMeasurementTrackerPA" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltPAFullTrackIter1ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter1Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter1Tau3MuSiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter1Tau3MuESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter1Tau3MuClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter1Tau3MuESPPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  ComponentName = cms.string( "hltIter1Tau3MuESPPixelLayerTriplets" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter1Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter1Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter1ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter1Tau3MuESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter1Tau3MuESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter2ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter2SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter2ESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter2ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter2ESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter2ESPPixelLayerPairs" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter2ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter2ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter2ESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter2ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter2ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltIter2ESPTrajectoryFilterIT" )
)
process.hltIter2Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter2SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter2Tau3MuESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter2Tau3MuClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter2Tau3MuESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter2Tau3MuESPPixelLayerPairs" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter2Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  ),
  TOB = cms.PSet(  )
)
process.hltIter2Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter2ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 2 ),
  ComponentName = cms.string( "hltIter2Tau3MuESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter2Tau3MuESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter3ESPLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter3ESPLayerTriplets" ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3ClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
process.hltIter3ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter3SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter3ESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter3ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter3ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter3ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter3ESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter3ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter3ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltIter3ESPTrajectoryFilterIT" )
)
process.hltIter3Tau3MuESPLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
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
  ComponentName = cms.string( "hltIter3Tau3MuESPLayerTriplets" ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0051 )
  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 ),
    useErrorsFromParam = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    skipClusters = cms.InputTag( "hltIter3Tau3MuClustersRefRemoval" ),
    hitErrorRPhi = cms.double( 0.0027 )
  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
process.hltIter3Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter3Tau3MuSiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter3Tau3MuESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter3Tau3MuClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter3Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter3ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter3Tau3MuESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter3Tau3MuESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.hltIter4ESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter4SiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter4ESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter4ClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter4ESPPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  layerList = cms.vstring( 'TIB1+TIB2' ),
  ComponentName = cms.string( "hltIter4ESPPixelLayerPairs" ),
  TEC = cms.PSet(  ),
  FPix = cms.PSet(  ),
  TID = cms.PSet(  ),
  BPix = cms.PSet(  ),
  TIB = cms.PSet(  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ) ),
  TOB = cms.PSet(  )
)
process.hltIter4ESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter4ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter4ESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter4ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  minNrOfHitsForRebuild = cms.untracked.int32( 4 )
)
process.hltIter4ESPTrajectoryFilterIT = cms.ESProducer( "TrajectoryFilterESProducer",
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
  ),
  ComponentName = cms.string( "hltIter4ESPTrajectoryFilterIT" )
)
process.hltIter4Tau3MuESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  OnDemand = cms.bool( True ),
  Regional = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  stripClusterProducer = cms.string( "hltIter4Tau3MuSiStripClusters" ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltIter4Tau3MuESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  skipClusters = cms.InputTag( "hltIter4Tau3MuClustersRefRemoval" ),
  UseStripModuleQualityDB = cms.bool( True ),
  UseStripNoiseDB = cms.bool( False ),
  UseStripCablingDB = cms.bool( False )
)
process.hltIter4Tau3MuESPTrajectoryBuilderIT = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilterName = cms.string( "hltIter4ESPTrajectoryFilterIT" ),
  maxCand = cms.int32( 1 ),
  ComponentName = cms.string( "hltIter4Tau3MuESPTrajectoryBuilderIT" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter4Tau3MuESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  minNrOfHitsForRebuild = cms.untracked.int32( 4 )
)
process.hoDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HODetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 30 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.muonDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "MuonDetIdAssociator" ),
  etaBinSize = cms.double( 0.125 ),
  nEta = cms.int32( 48 ),
  nPhi = cms.int32( 48 ),
  includeBadChambers = cms.bool( False )
)
process.navigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "SimpleNavigationSchool" )
)
process.preshowerDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "PreshowerDetIdAssociator" ),
  etaBinSize = cms.double( 0.1 ),
  nEta = cms.int32( 60 ),
  nPhi = cms.int32( 30 ),
  includeBadChambers = cms.bool( False )
)
process.siPixelQualityESProducer = cms.ESProducer( "SiPixelQualityESProducer",
  ListOfRecordToMerge = cms.VPSet( 
    cms.PSet(  record = cms.string( "SiPixelQualityFromDbRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiPixelDetVOffRcd" ),
      tag = cms.string( "" )
    )
  )
)
process.siPixelTemplateDBObjectESProducer = cms.ESProducer( "SiPixelTemplateDBObjectESProducer" )
process.siStripLorentzAngleDepESProducer = cms.ESProducer( "SiStripLorentzAngleDepESProducer",
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
process.sistripconn = cms.ESProducer( "SiStripConnectivity" )

process.FastTimerService = cms.Service( "FastTimerService",
    dqmPath = cms.untracked.string( "HLT/TimerService" ),
    dqmModuleTimeRange = cms.untracked.double( 40.0 ),
    luminosityProduct = cms.untracked.InputTag( "hltScalersRawToDigi" ),
    enableTimingExclusive = cms.untracked.bool( False ),
    enableTimingModules = cms.untracked.bool( True ),
    enableDQMbyPathOverhead = cms.untracked.bool( False ),
    dqmTimeResolution = cms.untracked.double( 5.0 ),
    enableDQMbyModule = cms.untracked.bool( False ),
    dqmLuminosityResolution = cms.untracked.double( 1.0E31 ),
    skipFirstPath = cms.untracked.bool( False ),
    enableTimingPaths = cms.untracked.bool( True ),
    enableDQMbyLumiSection = cms.untracked.bool( True ),
    dqmPathTimeResolution = cms.untracked.double( 0.5 ),
    dqmPathTimeRange = cms.untracked.double( 100.0 ),
    dqmTimeRange = cms.untracked.double( 1000.0 ),
    dqmLumiSectionsRange = cms.untracked.uint32( 2500 ),
    enableDQMSummary = cms.untracked.bool( True ),
    enableTimingSummary = cms.untracked.bool( False ),
    enableDQMbyPathTotal = cms.untracked.bool( False ),
    useRealTimeClock = cms.untracked.bool( True ),
    enableDQMbyPathExclusive = cms.untracked.bool( False ),
    enableDQMbyLuminosity = cms.untracked.bool( True ),
    enableDQM = cms.untracked.bool( True ),
    supportedProcesses = cms.untracked.vuint32( 8, 24, 32 ),
    dqmModuleTimeResolution = cms.untracked.double( 0.2 ),
    dqmLuminosityRange = cms.untracked.double( 1.0E34 ),
    enableDQMbyPathActive = cms.untracked.bool( False ),
    enableDQMbyPathDetails = cms.untracked.bool( False ),
    enableDQMbyProcesses = cms.untracked.bool( True ),
    enableDQMbyPathCounters = cms.untracked.bool( False )
)
process.DQM = cms.Service( "DQM",
    publishFrequency = cms.untracked.double( 5.0 ),
    debug = cms.untracked.bool( False ),
    collectorPort = cms.untracked.int32( 0 ),
    collectorHost = cms.untracked.string( "" )
)
process.DQMStore = cms.Service( "DQMStore",
)
process.DTDataIntegrityTask = cms.Service( "DTDataIntegrityTask",
    processingMode = cms.untracked.string( "HLT" ),
    fedIntegrityFolder = cms.untracked.string( "DT/FEDIntegrity_EvF" ),
    getSCInfo = cms.untracked.bool( True )
)
process.MessageLogger = cms.Service( "MessageLogger",
    suppressInfo = cms.untracked.vstring(  ),
    debugs = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      placeholder = cms.untracked.bool( True ),
      suppressInfo = cms.untracked.vstring(  ),
      suppressWarning = cms.untracked.vstring(  ),
      suppressDebug = cms.untracked.vstring(  ),
      suppressError = cms.untracked.vstring(  )
    ),
    suppressDebug = cms.untracked.vstring(  ),
    cout = cms.untracked.PSet( 
      threshold = cms.untracked.string( "ERROR" ),
      suppressInfo = cms.untracked.vstring(  ),
      suppressWarning = cms.untracked.vstring(  ),
      suppressDebug = cms.untracked.vstring(  ),
      suppressError = cms.untracked.vstring(  )
    ),
    cerr_stats = cms.untracked.PSet( 
      threshold = cms.untracked.string( "WARNING" ),
      output = cms.untracked.string( "cerr" ),
      optionalPSet = cms.untracked.bool( True )
    ),
    warnings = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      placeholder = cms.untracked.bool( True ),
      suppressInfo = cms.untracked.vstring(  ),
      suppressWarning = cms.untracked.vstring(  ),
      suppressDebug = cms.untracked.vstring(  ),
      suppressError = cms.untracked.vstring(  )
    ),
    statistics = cms.untracked.vstring( 'cerr' ),
    cerr = cms.untracked.PSet( 
      INFO = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      noTimeStamps = cms.untracked.bool( False ),
      FwkReport = cms.untracked.PSet( 
        reportEvery = cms.untracked.int32( 1 ),
        limit = cms.untracked.int32( 0 )
      ),
      default = cms.untracked.PSet(  limit = cms.untracked.int32( 10000000 ) ),
      Root_NoDictionary = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      FwkJob = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      FwkSummary = cms.untracked.PSet( 
        reportEvery = cms.untracked.int32( 1 ),
        limit = cms.untracked.int32( 10000000 )
      ),
      threshold = cms.untracked.string( "INFO" ),
      suppressInfo = cms.untracked.vstring(  ),
      suppressWarning = cms.untracked.vstring(  ),
      suppressDebug = cms.untracked.vstring(  ),
      suppressError = cms.untracked.vstring(  )
    ),
    FrameworkJobReport = cms.untracked.PSet( 
      default = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      FwkJob = cms.untracked.PSet(  limit = cms.untracked.int32( 10000000 ) )
    ),
    suppressWarning = cms.untracked.vstring( 'hltOnlineBeamSpot',
      'hltCtf3HitL1SeededWithMaterialTracks',
      'hltL3MuonsOIState',
      'hltPixelTracksForHighMult',
      'hltHITPixelTracksHE',
      'hltHITPixelTracksHB',
      'hltCtfL1SeededWithMaterialTracks',
      'hltRegionalTracksForL3MuonIsolation',
      'hltSiPixelClusters',
      'hltActivityStartUpElectronPixelSeeds',
      'hltLightPFTracks',
      'hltPixelVertices3DbbPhi',
      'hltL3MuonsIOHit',
      'hltPixelTracks',
      'hltSiPixelDigis',
      'hltL3MuonsOIHit',
      'hltL1SeededElectronGsfTracks',
      'hltL1SeededStartUpElectronPixelSeeds',
      'hltBLifetimeRegionalCtfWithMaterialTracksbbPhiL1FastJetFastPV',
      'hltCtfActivityWithMaterialTracks' ),
    errors = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      placeholder = cms.untracked.bool( True ),
      suppressInfo = cms.untracked.vstring(  ),
      suppressWarning = cms.untracked.vstring(  ),
      suppressDebug = cms.untracked.vstring(  ),
      suppressError = cms.untracked.vstring(  )
    ),
    fwkJobReports = cms.untracked.vstring( 'FrameworkJobReport' ),
    debugModules = cms.untracked.vstring(  ),
    infos = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      Root_NoDictionary = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      placeholder = cms.untracked.bool( True ),
      suppressInfo = cms.untracked.vstring(  ),
      suppressWarning = cms.untracked.vstring(  ),
      suppressDebug = cms.untracked.vstring(  ),
      suppressError = cms.untracked.vstring(  )
    ),
    categories = cms.untracked.vstring( 'FwkJob',
      'FwkReport',
      'FwkSummary',
      'Root_NoDictionary' ),
    destinations = cms.untracked.vstring( 'warnings',
      'errors',
      'infos',
      'debugs',
      'cout',
      'cerr' ),
    threshold = cms.untracked.string( "INFO" ),
    suppressError = cms.untracked.vstring( 'hltOnlineBeamSpot',
      'hltL3MuonCandidates',
      'hltL3TkTracksFromL2OIState',
      'hltPFJetCtfWithMaterialTracks',
      'hltL3TkTracksFromL2IOHit',
      'hltL3TkTracksFromL2OIHit' )
)
process.MicroStateService = cms.Service( "MicroStateService",
)
process.ModuleWebRegistry = cms.Service( "ModuleWebRegistry",
)
process.PrescaleService = cms.Service( "PrescaleService",
    forceDefault = cms.bool( False ),
    prescaleTable = cms.VPSet( 
      cms.PSet(  pathName = cms.string( "HLT_HIMET120_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMET200_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMET220_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIPhysics_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 600, 500, 400, 300, 200, 150, 100, 50, 200, 200, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIDTCalibration_v3" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIEcalCalibration_v3" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIHcalCalibration_v3" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIZeroBias_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIZeroBiasXOR_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIZeroBiasPixel_SingleTrack_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMinBiasBSC_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMinBiasBSC_OR_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMinBiasHF_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMinBiasHf_OR_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMinBiasHfOrBSC_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMinBiasPixel_SingleTrack_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMinBiasZDC_Calo_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMinBiasZDCPixel_SingleTrack_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIBptxXOR_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL1Algo_BptxXOR_BSC_OR_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL1DoubleMuOpen_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL1DoubleMu0_HighQ_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL2Mu3_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL2Mu3_NHitQ_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 7, 5, 3, 1, 1, 1, 1, 1, 7, 3, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL2Mu7_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL2Mu15_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL2DoubleMu0_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL2DoubleMu0_NHitQ_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL2DoubleMu3_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL3Mu3_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 25, 15, 10, 1, 1, 1, 1, 1, 35, 35, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL3DoubleMuOpen_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 7, 5, 3, 1, 1, 1, 1, 1, 5, 5, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL3DoubleMuOpen_Mgt2_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL3DoubleMuOpen_Mgt2_SS_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL3DoubleMuOpen_Mgt2_OS_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HISinglePhoton15_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 17, 13, 11, 7, 5, 3, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HISinglePhoton20_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 1, 1, 1, 1, 1, 3, 3, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HISinglePhoton30_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HISinglePhoton40_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIPhoton10_Photon15_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 33, 23, 17, 7, 3, 2, 1, 1, 1, 30, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIPhoton15_Photon20_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIDoublePhoton10_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 60, 50, 40, 30, 20, 10, 5, 2, 300, 300, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIDoublePhoton15_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIDoublePhoton20_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIJet55_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 5, 4, 3, 1, 1, 1, 5, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIJet65_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIJet80_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIJet95_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIDiJet55_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIJet65_Jet55_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIJetE30_NoBPTX_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIJetE50_NoBPTX3BX_NoHalo_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIActivityHF_Coincidence3_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIActivityHF_Single3_v4" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIClusterVertexCompatibility_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HICentralityVeto_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIFullTrack12_L1Central_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 3, 5, 4, 2, 1, 1, 1, 1, 5, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIFullTrack12_L1Peripheral_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIFullTrack14_L1Central_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIFullTrack14_L1Peripheral_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIFullTrack20_L1Central_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIFullTrack20_L1Peripheral_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIFullTrack25_L1Central_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIFullTrack25_L1Peripheral_v6" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIRandom_v3" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIUCC010_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIUCC015_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HICentral10_v7" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 40, 35, 30, 25, 15, 12, 7, 4, 25, 25, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIUPCNeuMuPixel_SingleTrack_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIUPCNeuEG2Pixel_SingleTrack_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIUPCNeuEG5Pixel_SingleTrack_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "AForHIOutput" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "ALCALUMIPIXELSOutput" ),
        prescales = cms.vuint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "DQMOutput" ),
        prescales = cms.vuint32( 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 1, 1 )
      ),
      cms.PSet(  pathName = cms.string( "ExpressForHIOutput" ),
        prescales = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 )
      ),
      cms.PSet(  pathName = cms.string( "HLTMONOutput" ),
        prescales = cms.vuint32( 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 )
      )
    ),
    lvl1DefaultLabel = cms.string( "3e33" ),
    lvl1Labels = cms.vstring( '9e33nopark',
      '8e33nopark',
      '8e33',
      '7e33',
      '6e33',
      '4e33',
      '2e33',
      '5e32',
      '6000Hz',
      '5000Hz',
      '4000Hz',
      '3000Hz',
      '2000Hz',
      '1500Hz',
      '1000Hz',
      '500Hz',
      'EM1',
      'EM2',
      'PAPilot8Bunches',
      'PAPilot8BunchesEM',
      'PAPilot4Bunches',
      'PAPilot4BunchesEM',
      'CirculatingBeam',
      'CirculatingBeam+HighRandom' )
)
process.UpdaterService = cms.Service( "UpdaterService",
)

process.hltGetConditions = cms.EDAnalyzer( "EventSetupRecordDataGetter",
    toGet = cms.VPSet( 
    ),
    verbose = cms.untracked.bool( False )
)
process.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
process.hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
process.hltTriggerType = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 1 )
)
process.hltGtDigis = cms.EDProducer( "L1GlobalTriggerRawToDigi",
    DaqGtFedId = cms.untracked.int32( 813 ),
    DaqGtInputTag = cms.InputTag( "rawDataCollector" ),
    UnpackBxInEvent = cms.int32( 5 ),
    ActiveBoardsMask = cms.uint32( 0xffff )
)
process.hltGctDigis = cms.EDProducer( "GctRawToDigi",
    unpackSharedRegions = cms.bool( False ),
    numberOfGctSamplesToUnpack = cms.uint32( 1 ),
    verbose = cms.untracked.bool( False ),
    numberOfRctSamplesToUnpack = cms.uint32( 1 ),
    inputLabel = cms.InputTag( "rawDataCollector" ),
    unpackerVersion = cms.uint32( 0 ),
    gctFedId = cms.untracked.int32( 745 ),
    hltMode = cms.bool( True )
)
process.hltL1GtObjectMap = cms.EDProducer( "L1GlobalTrigger",
    TechnicalTriggersUnprescaled = cms.bool( True ),
    ProduceL1GtObjectMapRecord = cms.bool( True ),
    AlgorithmTriggersUnmasked = cms.bool( False ),
    EmulateBxInEvent = cms.int32( 1 ),
    AlgorithmTriggersUnprescaled = cms.bool( True ),
    ProduceL1GtDaqRecord = cms.bool( False ),
    ReadTechnicalTriggerRecords = cms.bool( True ),
    RecordLength = cms.vint32( 3, 0 ),
    TechnicalTriggersUnmasked = cms.bool( False ),
    ProduceL1GtEvmRecord = cms.bool( False ),
    GmtInputTag = cms.InputTag( "hltGtDigis" ),
    TechnicalTriggersVetoUnmasked = cms.bool( True ),
    AlternativeNrBxBoardEvm = cms.uint32( 0 ),
    TechnicalTriggersInputTags = cms.VInputTag( 'simBscDigis' ),
    CastorInputTag = cms.InputTag( "castorL1Digis" ),
    GctInputTag = cms.InputTag( "hltGctDigis" ),
    AlternativeNrBxBoardDaq = cms.uint32( 0 ),
    WritePsbL1GtDaqRecord = cms.bool( False ),
    BstLengthBytes = cms.int32( -1 )
)
process.hltL1extraParticles = cms.EDProducer( "L1ExtraParticlesProd",
    tauJetSource = cms.InputTag( 'hltGctDigis','tauJets' ),
    etHadSource = cms.InputTag( "hltGctDigis" ),
    etTotalSource = cms.InputTag( "hltGctDigis" ),
    centralBxOnly = cms.bool( True ),
    centralJetSource = cms.InputTag( 'hltGctDigis','cenJets' ),
    etMissSource = cms.InputTag( "hltGctDigis" ),
    hfRingEtSumsSource = cms.InputTag( "hltGctDigis" ),
    produceMuonParticles = cms.bool( True ),
    forwardJetSource = cms.InputTag( 'hltGctDigis','forJets' ),
    ignoreHtMiss = cms.bool( False ),
    htMissSource = cms.InputTag( "hltGctDigis" ),
    produceCaloParticles = cms.bool( True ),
    muonSource = cms.InputTag( "hltGtDigis" ),
    isolatedEmSource = cms.InputTag( 'hltGctDigis','isoEm' ),
    nonIsolatedEmSource = cms.InputTag( 'hltGctDigis','nonIsoEm' ),
    hfRingBitCountsSource = cms.InputTag( "hltGctDigis" )
)
process.hltScalersRawToDigi = cms.EDProducer( "ScalersRawToDigi",
    scalersInputTag = cms.InputTag( "rawDataCollector" )
)
process.hltOnlineBeamSpot = cms.EDProducer( "BeamSpotOnlineProducer",
    maxZ = cms.double( 40.0 ),
    src = cms.InputTag( "hltScalersRawToDigi" ),
    gtEvmLabel = cms.InputTag( "" ),
    changeToCMSCoordinates = cms.bool( False ),
    setSigmaZ = cms.double( 0.0 ),
    maxRadius = cms.double( 2.0 )
)
process.hltL1sL1ETM30BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM30_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIMET120 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEcalRawToRecHitFacility = cms.EDProducer( "EcalRawToRecHitFacility",
    sourceTag = cms.InputTag( "rawDataCollector" ),
    workerName = cms.string( "" )
)
process.hltEcalRegionalRestFEDs = cms.EDProducer( "EcalRawToRecHitRoI",
    JetJobPSet = cms.VPSet( 
    ),
    sourceTag_es = cms.InputTag( "NotNeededoESfalse" ),
    doES = cms.bool( False ),
    type = cms.string( "all" ),
    sourceTag = cms.InputTag( "hltEcalRawToRecHitFacility" ),
    EmJobPSet = cms.VPSet( 
    ),
    CandJobPSet = cms.VPSet( 
    ),
    MuonJobPSet = cms.PSet(  ),
    esInstance = cms.untracked.string( "es" ),
    MuJobPSet = cms.PSet(  )
)
process.hltEcalRecHitAll = cms.EDProducer( "EcalRawToRecHitProducer",
    splitOutput = cms.bool( True ),
    rechitCollection = cms.string( "NotNeededsplitOutputTrue" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    sourceTag = cms.InputTag( "hltEcalRegionalRestFEDs" ),
    cleaningConfig = cms.PSet( 
      e6e2thresh = cms.double( 0.04 ),
      tightenCrack_e6e2_double = cms.double( 3.0 ),
      e4e1Threshold_endcap = cms.double( 0.3 ),
      tightenCrack_e4e1_single = cms.double( 3.0 ),
      tightenCrack_e1_double = cms.double( 2.0 ),
      cThreshold_barrel = cms.double( 4.0 ),
      e4e1Threshold_barrel = cms.double( 0.08 ),
      tightenCrack_e1_single = cms.double( 2.0 ),
      e4e1_b_barrel = cms.double( -0.024 ),
      e4e1_a_barrel = cms.double( 0.04 ),
      ignoreOutOfTimeThresh = cms.double( 1.0E9 ),
      cThreshold_endcap = cms.double( 15.0 ),
      e4e1_b_endcap = cms.double( -0.0125 ),
      e4e1_a_endcap = cms.double( 0.02 ),
      cThreshold_double = cms.double( 10.0 )
    ),
    lazyGetterTag = cms.InputTag( "hltEcalRawToRecHitFacility" )
)
process.hltHcalDigis = cms.EDProducer( "HcalRawToDigi",
    UnpackZDC = cms.untracked.bool( True ),
    FilterDataQuality = cms.bool( True ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    ComplainEmptyData = cms.untracked.bool( False ),
    UnpackCalib = cms.untracked.bool( True ),
    UnpackTTP = cms.untracked.bool( False ),
    lastSample = cms.int32( 9 ),
    firstSample = cms.int32( 0 )
)
process.hltHbhereco = cms.EDProducer( "HcalHitReconstructor",
    digiTimeFromDB = cms.bool( True ),
    S9S1stat = cms.PSet(  ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    tsFromDB = cms.bool( True ),
    samplesToAdd = cms.int32( 4 ),
    correctionPhaseNS = cms.double( 13.0 ),
    HFInWindowStat = cms.PSet(  ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    setHSCPFlags = cms.bool( False ),
    firstAuxTS = cms.int32( 4 ),
    setSaturationFlags = cms.bool( False ),
    hfTimingTrustParameters = cms.PSet(  ),
    PETstat = cms.PSet(  ),
    digistat = cms.PSet(  ),
    useLeakCorrection = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    S8S1stat = cms.PSet(  ),
    correctForPhaseContainment = cms.bool( True ),
    correctForTimeslew = cms.bool( True ),
    setNoiseFlags = cms.bool( False ),
    correctTiming = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    Subdetector = cms.string( "HBHE" ),
    dropZSmarkedPassed = cms.bool( True ),
    recoParamsFromDB = cms.bool( True ),
    firstSample = cms.int32( 4 ),
    setTimingShapedCutsFlags = cms.bool( False ),
    timingshapedcutsParameters = cms.PSet( 
      ignorelowest = cms.bool( True ),
      win_offset = cms.double( 0.0 ),
      ignorehighest = cms.bool( False ),
      win_gain = cms.double( 1.0 ),
      tfilterEnvelope = cms.vdouble( 4.0, 12.04, 13.0, 10.56, 23.5, 8.82, 37.0, 7.38, 56.0, 6.3, 81.0, 5.64, 114.5, 5.44, 175.5, 5.38, 350.5, 5.14 )
    ),
    pulseShapeParameters = cms.PSet(  ),
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
    )
)
process.hltHfreco = cms.EDProducer( "HcalHitReconstructor",
    digiTimeFromDB = cms.bool( True ),
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
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    tsFromDB = cms.bool( True ),
    samplesToAdd = cms.int32( 2 ),
    correctionPhaseNS = cms.double( 13.0 ),
    HFInWindowStat = cms.PSet( 
      hflongEthresh = cms.double( 40.0 ),
      hflongMinWindowTime = cms.vdouble( -10.0 ),
      hfshortEthresh = cms.double( 40.0 ),
      hflongMaxWindowTime = cms.vdouble( 10.0 ),
      hfshortMaxWindowTime = cms.vdouble( 10.0 ),
      hfshortMinWindowTime = cms.vdouble( -12.0 )
    ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    setHSCPFlags = cms.bool( False ),
    firstAuxTS = cms.int32( 1 ),
    setSaturationFlags = cms.bool( False ),
    hfTimingTrustParameters = cms.PSet( 
      hfTimingTrustLevel2 = cms.int32( 4 ),
      hfTimingTrustLevel1 = cms.int32( 1 )
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
    digistat = cms.PSet( 
      HFdigiflagFirstSample = cms.int32( 1 ),
      HFdigiflagMinEthreshold = cms.double( 40.0 ),
      HFdigiflagSamplesToAdd = cms.int32( 3 ),
      HFdigiflagExpectedPeak = cms.int32( 2 ),
      HFdigiflagCoef = cms.vdouble( 0.93, -0.012667, -0.38275 )
    ),
    useLeakCorrection = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
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
    correctForPhaseContainment = cms.bool( False ),
    correctForTimeslew = cms.bool( False ),
    setNoiseFlags = cms.bool( True ),
    correctTiming = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    Subdetector = cms.string( "HF" ),
    dropZSmarkedPassed = cms.bool( True ),
    recoParamsFromDB = cms.bool( True ),
    firstSample = cms.int32( 2 ),
    setTimingShapedCutsFlags = cms.bool( False ),
    timingshapedcutsParameters = cms.PSet(  ),
    pulseShapeParameters = cms.PSet(  ),
    flagParameters = cms.PSet(  ),
    hscpParameters = cms.PSet(  )
)
process.hltHoreco = cms.EDProducer( "HcalHitReconstructor",
    digiTimeFromDB = cms.bool( True ),
    S9S1stat = cms.PSet(  ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    tsFromDB = cms.bool( True ),
    samplesToAdd = cms.int32( 4 ),
    correctionPhaseNS = cms.double( 13.0 ),
    HFInWindowStat = cms.PSet(  ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    setHSCPFlags = cms.bool( False ),
    firstAuxTS = cms.int32( 4 ),
    setSaturationFlags = cms.bool( False ),
    hfTimingTrustParameters = cms.PSet(  ),
    PETstat = cms.PSet(  ),
    digistat = cms.PSet(  ),
    useLeakCorrection = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    S8S1stat = cms.PSet(  ),
    correctForPhaseContainment = cms.bool( True ),
    correctForTimeslew = cms.bool( True ),
    setNoiseFlags = cms.bool( False ),
    correctTiming = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    Subdetector = cms.string( "HO" ),
    dropZSmarkedPassed = cms.bool( True ),
    recoParamsFromDB = cms.bool( True ),
    firstSample = cms.int32( 4 ),
    setTimingShapedCutsFlags = cms.bool( False ),
    timingshapedcutsParameters = cms.PSet(  ),
    pulseShapeParameters = cms.PSet(  ),
    flagParameters = cms.PSet(  ),
    hscpParameters = cms.PSet(  )
)
process.hltTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
    EBSumThreshold = cms.double( 0.2 ),
    MomHBDepth = cms.double( 0.2 ),
    UseEtEBTreshold = cms.bool( False ),
    hfInput = cms.InputTag( "hltHfreco" ),
    AllowMissingInputs = cms.bool( False ),
    MomEEDepth = cms.double( 0.0 ),
    EESumThreshold = cms.double( 0.45 ),
    HBGrid = cms.vdouble(  ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    HBThreshold = cms.double( 0.7 ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  ),
    UseEcalRecoveredHits = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHEDepth = cms.double( 0.4 ),
    HcalThreshold = cms.double( -1000.0 ),
    HF2Weights = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    UseSymEBTreshold = cms.bool( False ),
    EEWeights = cms.vdouble(  ),
    EEWeight = cms.double( 1.0 ),
    UseHO = cms.bool( False ),
    HBWeights = cms.vdouble(  ),
    HF1Weight = cms.double( 1.0 ),
    HF2Grid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    EBWeight = cms.double( 1.0 ),
    HF1Grid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    HOWeight = cms.double( 1.0E-99 ),
    HESWeight = cms.double( 1.0 ),
    HESThreshold = cms.double( 0.8 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    HF2Weight = cms.double( 1.0 ),
    HF2Threshold = cms.double( 0.85 ),
    HcalAcceptSeverityLevel = cms.uint32( 9 ),
    EEThreshold = cms.double( 0.3 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HF1Weights = cms.vdouble(  ),
    hoInput = cms.InputTag( "hltHoreco" ),
    HF1Threshold = cms.double( 0.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HESGrid = cms.vdouble(  ),
    EcutTower = cms.double( -1000.0 ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    HESWeights = cms.vdouble(  ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kTime',
      'kWeird',
      'kBad' ),
    HEDWeight = cms.double( 1.0 ),
    UseSymEETreshold = cms.bool( False ),
    HEDThreshold = cms.double( 0.8 ),
    EBThreshold = cms.double( 0.07 ),
    UseRejectedHitsOnly = cms.bool( False ),
    UseHcalRecoveredHits = cms.bool( False ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HOThreshold0 = cms.double( 3.5 ),
    ecalInputs = cms.VInputTag( 'hltEcalRecHitAll:EcalRecHitsEB','hltEcalRecHitAll:EcalRecHitsEE' ),
    UseRejectedRecoveredHcalHits = cms.bool( False ),
    MomEBDepth = cms.double( 0.3 ),
    HBWeight = cms.double( 1.0 ),
    HOGrid = cms.vdouble(  ),
    EBGrid = cms.vdouble(  )
)
process.hltMet = cms.EDProducer( "METProducer",
    resolutionsEra = cms.string( "Spring10" ),
    HB_EtResPar = cms.vdouble( 0.0, 1.22, 0.05 ),
    EE_PhiResPar = cms.vdouble( 0.02511 ),
    jdpt9 = cms.vdouble( 0.843, 0.885, 1.245, 1.665, 1.944, 1.981, 1.972, 2.875, 3.923, 7.51 ),
    jdpt8 = cms.vdouble( 0.889, 0.939, 1.166, 1.365, 1.553, 1.805, 2.06, 2.22, 2.268, 2.247 ),
    jdpt7 = cms.vdouble( 1.094, 1.139, 1.436, 1.672, 1.831, 2.05, 2.267, 2.549, 2.785, 2.86 ),
    jdpt6 = cms.vdouble( 1.213, 1.298, 1.716, 2.015, 2.191, 2.612, 2.863, 2.879, 2.925, 2.902 ),
    jdpt5 = cms.vdouble( 1.049, 1.149, 1.607, 1.869, 2.012, 2.219, 2.289, 2.412, 2.695, 2.865 ),
    jdpt4 = cms.vdouble( 0.85, 0.961, 1.337, 1.593, 1.854, 2.005, 2.209, 2.533, 2.812, 3.047 ),
    jdpt3 = cms.vdouble( 0.929, 1.04, 1.46, 1.74, 2.042, 2.289, 2.639, 2.837, 2.946, 2.971 ),
    jdpt2 = cms.vdouble( 0.841, 0.937, 1.316, 1.605, 1.919, 2.295, 2.562, 2.722, 2.943, 3.293 ),
    jdpt1 = cms.vdouble( 0.718, 0.813, 1.133, 1.384, 1.588, 1.841, 2.115, 2.379, 2.508, 2.772 ),
    jdpt0 = cms.vdouble( 0.749, 0.829, 1.099, 1.355, 1.584, 1.807, 2.035, 2.217, 2.378, 2.591 ),
    HE_EtResPar = cms.vdouble( 0.0, 1.3, 0.05 ),
    alias = cms.string( "RawCaloMET" ),
    HF_PhiResPar = cms.vdouble( 0.05022 ),
    InputType = cms.string( "CandidateCollection" ),
    HE_PhiResPar = cms.vdouble( 0.02511 ),
    HB_PhiResPar = cms.vdouble( 0.02511 ),
    EE_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    noHF = cms.bool( False ),
    PF_PhiResType2 = cms.vdouble( 0.0020 ),
    PF_PhiResType3 = cms.vdouble( 0.0020 ),
    HF_EtResPar = cms.vdouble( 0.0, 1.82, 0.09 ),
    resolutionsAlgo = cms.string( "AK5PF" ),
    PF_PhiResType6 = cms.vdouble( 0.02511 ),
    PF_PhiResType7 = cms.vdouble( 0.02511 ),
    PF_PhiResType4 = cms.vdouble( 0.0028, 0.0, 0.0022 ),
    PF_PhiResType5 = cms.vdouble( 0.1, 0.1, 0.13 ),
    ptresolthreshold = cms.double( 10.0 ),
    METType = cms.string( "CaloMET" ),
    EB_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    PF_PhiResType1 = cms.vdouble( 0.0020 ),
    globalThreshold = cms.double( 0.3 ),
    EB_PhiResPar = cms.vdouble( 0.00502 ),
    src = cms.InputTag( "hltTowerMakerForAll" ),
    jdphi9 = cms.vdouble( 0.062, 0.059, 0.053, 0.047, 0.042, 0.045, 0.036, 0.032, 0.034, 0.044 ),
    jdphi8 = cms.vdouble( 0.059, 0.057, 0.051, 0.044, 0.038, 0.035, 0.037, 0.032, 0.028, 0.028 ),
    jdphi4 = cms.vdouble( 0.042, 0.042, 0.043, 0.042, 0.038, 0.036, 0.036, 0.033, 0.031, 0.031 ),
    jdphi3 = cms.vdouble( 0.042, 0.043, 0.044, 0.043, 0.041, 0.039, 0.039, 0.036, 0.034, 0.031 ),
    jdphi2 = cms.vdouble( 0.04, 0.04, 0.04, 0.04, 0.04, 0.038, 0.036, 0.035, 0.034, 0.033 ),
    jdphi1 = cms.vdouble( 0.034, 0.035, 0.035, 0.035, 0.035, 0.034, 0.031, 0.03, 0.029, 0.027 ),
    jdphi0 = cms.vdouble( 0.034, 0.034, 0.034, 0.034, 0.032, 0.031, 0.028, 0.027, 0.027, 0.027 ),
    jdphi7 = cms.vdouble( 0.077, 0.072, 0.059, 0.05, 0.045, 0.042, 0.039, 0.039, 0.037, 0.031 ),
    jdphi6 = cms.vdouble( 0.084, 0.08, 0.072, 0.065, 0.066, 0.06, 0.051, 0.049, 0.045, 0.045 ),
    jdphi5 = cms.vdouble( 0.069, 0.069, 0.064, 0.058, 0.053, 0.049, 0.049, 0.043, 0.039, 0.04 ),
    HO_EtResPar = cms.vdouble( 0.0, 1.3, 0.0050 ),
    HO_PhiResPar = cms.vdouble( 0.02511 ),
    PF_EtResType5 = cms.vdouble( 0.41, 0.52, 0.25 ),
    PF_EtResType4 = cms.vdouble( 0.042, 0.1, 0.0 ),
    PF_EtResType7 = cms.vdouble( 0.0, 1.22, 0.05 ),
    PF_EtResType6 = cms.vdouble( 0.0, 1.22, 0.05 ),
    PF_EtResType1 = cms.vdouble( 0.05, 0.0, 0.0 ),
    calculateSignificance = cms.bool( False ),
    PF_EtResType3 = cms.vdouble( 0.05, 0.0, 0.0 ),
    PF_EtResType2 = cms.vdouble( 0.05, 0.0, 0.0 ),
    usePt = cms.untracked.bool( False ),
    onlyFiducialParticles = cms.bool( False ),
    vertexRho = cms.double( 2.0 ),
    eVetoDeltaPhi = cms.double( 100.0 ),
    PFClustersHCAL = cms.InputTag( "particleFlowClusterHCAL" ),
    PFClustersHFHAD = cms.InputTag( "particleFlowClusterHFHAD" ),
    dupMinPt = cms.double( 0.0 ),
    chi2_tight_max = cms.double( 5.0 ),
    vertexZ = cms.double( 15.0 ),
    nLayersTight = cms.int32( 0 ),
    vertexNdof = cms.int32( 4 ),
    ptErr_max = cms.double( 0.2 ),
    corner = cms.double( 1.479 ),
    PFClustersECAL = cms.InputTag( "particleFlowClusterECAL" ),
    eta_max = cms.double( 2.65 ),
    muonInputTag = cms.InputTag( "muons" ),
    eVetoDeltaCotTheta = cms.double( 100.0 ),
    maxd0cut = cms.double( 0.3 ),
    PFClustersHFEM = cms.InputTag( "particleFlowClusterHFEM" ),
    d0cutb = cms.double( 0.5 ),
    checkTrackPropagation = cms.bool( False ),
    usePFClusters = cms.bool( False ),
    vertexMaxDZ = cms.double( 0.2 ),
    deltaRShower = cms.double( 0.01 ),
    chi2_max = cms.double( 5.0 ),
    maxpt_eta25 = cms.double( 0.0 ),
    track_algos = cms.vint32(  ),
    ptErr_tight_max = cms.double( 0.2 ),
    maxTrackAlgo = cms.int32( 8 ),
    nLayers = cms.int32( 0 ),
    correctShowerTracks = cms.bool( False ),
    vetoDuplicates = cms.bool( False ),
    pt_max = cms.double( 100.0 ),
    radius = cms.double( 130.0 ),
    nhits_tight_min = cms.double( 9.0 ),
    beamSpotInputTag = cms.InputTag( "unused" ),
    dupDCotTh = cms.double( 6.0E-4 ),
    usedeltaRRejection = cms.bool( False ),
    trackInputTag = cms.InputTag( "generalTracks" ),
    dupDPhi = cms.double( 0.03 ),
    electronInputTag = cms.InputTag( "gsfElectrons" ),
    tcmetDepValueMap = cms.InputTag( 'muonTCMETValueMapProducer','muCorrData' ),
    d0cuta = cms.double( 0.015 ),
    hOverECut = cms.double( 0.1 ),
    electronVetoCone = cms.bool( True ),
    muonDepValueMap = cms.InputTag( 'muonMETValueMapProducer','muCorrData' ),
    metInputTag = cms.InputTag( "met" ),
    usePvtxd0 = cms.bool( False ),
    vertexInputTag = cms.InputTag( "offlinePrimaryVertices" ),
    zdist = cms.double( 314.0 ),
    nhits_min = cms.double( 6.0 ),
    eVetoDeltaR = cms.double( 0.015 ),
    maxpt_eta20 = cms.double( 100.0 ),
    pt_min = cms.double( 1.0 ),
    rf_type = cms.int32( 0 ),
    nMinOuterHits = cms.int32( 2 ),
    track_quality = cms.vint32( 2 ),
    isCosmics = cms.bool( False ),
    eVetoMinElectronPt = cms.double( 10.0 )
)
process.hltHIMET120 = cms.EDFilter( "HLT1CaloMET",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 120.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( -1.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltMet" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 87 )
)
process.hltSiStripRawToDigi = cms.EDProducer( "SiStripRawToDigiModule",
    UseDaqRegister = cms.bool( False ),
    ProductLabel = cms.InputTag( "rawDataCollector" ),
    MarkModulesOnMissingFeds = cms.bool( True ),
    UnpackCommonModeValues = cms.bool( False ),
    AppendedBytes = cms.int32( 0 ),
    UseFedKey = cms.bool( False ),
    ErrorThreshold = cms.uint32( 7174 ),
    TriggerFedId = cms.int32( 0 ),
    DoAPVEmulatorCheck = cms.bool( False ),
    UnpackBadChannels = cms.bool( False ),
    DoAllCorruptBufferChecks = cms.bool( False )
)
process.hltSiStripZeroSuppression = cms.EDProducer( "SiStripZeroSuppression",
    fixCM = cms.bool( False ),
    DigisToMergeVR = cms.InputTag( 'siStripVRDigis','VirginRaw' ),
    produceCalculatedBaseline = cms.bool( False ),
    produceBaselinePoints = cms.bool( False ),
    RawDigiProducersList = cms.VInputTag( 'hltSiStripRawToDigi:VirginRaw','hltSiStripRawToDigi:ProcessedRaw','hltSiStripRawToDigi:ScopeMode' ),
    storeInZScollBadAPV = cms.bool( True ),
    mergeCollections = cms.bool( False ),
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
    DigisToMergeZS = cms.InputTag( 'siStripDigis','ZeroSuppressed' ),
    storeCM = cms.bool( False ),
    produceRawDigis = cms.bool( True )
)
process.hltSiStripDigiToZSRaw = cms.EDProducer( "SiStripDigiToRawModule",
    InputDigiLabel = cms.string( "VirginRaw" ),
    FedReadoutMode = cms.string( "ZERO_SUPPRESSED" ),
    UseWrongDigiType = cms.bool( False ),
    UseFedKey = cms.bool( False ),
    InputModuleLabel = cms.string( "hltSiStripZeroSuppression" )
)
process.hltSiStripRawDigiToVirginRaw = cms.EDProducer( "SiStripDigiToRawModule",
    InputDigiLabel = cms.string( "VirginRaw" ),
    FedReadoutMode = cms.string( "VIRGIN_RAW" ),
    UseWrongDigiType = cms.bool( False ),
    UseFedKey = cms.bool( False ),
    InputModuleLabel = cms.string( "hltSiStripZeroSuppression" )
)
process.virginRawDataRepacker = cms.EDProducer( "RawDataCollectorByLabel",
    verbose = cms.untracked.int32( 0 ),
    RawCollectionList = cms.VInputTag( 'hltSiStripRawDigiToVirginRaw' )
)
process.rawDataRepacker = cms.EDProducer( "RawDataCollectorByLabel",
    verbose = cms.untracked.int32( 0 ),
    RawCollectionList = cms.VInputTag( 'hltSiStripDigiToZSRaw','source','rawDataCollector' )
)
process.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
process.hltL1sL1ETM50BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM50_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIMET200 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIMET200 = cms.EDFilter( "HLT1CaloMET",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 200.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( -1.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltMet" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 87 )
)
process.hltPreHIMET220 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIMET220 = cms.EDFilter( "HLT1CaloMET",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 220.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( -1.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltMet" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 87 )
)
process.hltPreHIPhysics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltCalibrationEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 2 )
)
process.hltPreHIDTCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltDTCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780 )
)
process.hltPreHIEcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 )
)
process.hltPreHIHcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHcalCalibTypeFilter = cms.EDFilter( "HLTHcalCalibTypeFilter",
    InputTag = cms.InputTag( "rawDataCollector" ),
    CalibTypes = cms.vint32( 1, 2, 3, 4, 5, 6 ),
    FilterSummary = cms.untracked.bool( False )
)
process.hltHcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731 )
)
process.hltL1sHIZeroBias = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1BptxXOR = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxXOR" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIZeroBiasXOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIZeroBiasXOR = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxPlusANDMinus OR L1_BptxXOR" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIZeroBiasPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSiPixelDigis = cms.EDProducer( "SiPixelRawToDigi",
    UseQualityInfo = cms.bool( False ),
    CheckPixelOrder = cms.bool( False ),
    IncludeErrors = cms.bool( False ),
    UseCablingTree = cms.untracked.bool( True ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    ErrorList = cms.vint32(  ),
    Regions = cms.PSet(  ),
    Timing = cms.untracked.bool( False ),
    UserErrorList = cms.vint32(  )
)
process.hltHISiPixelClusters = cms.EDProducer( "SiPixelClusterProducer",
    src = cms.InputTag( "hltSiPixelDigis" ),
    ChannelThreshold = cms.int32( 1000 ),
    maxNumberOfClusters = cms.int32( -1 ),
    VCaltoElectronGain = cms.int32( 65 ),
    MissCalibrate = cms.untracked.bool( True ),
    SplitClusters = cms.bool( False ),
    VCaltoElectronOffset = cms.int32( -414 ),
    payloadType = cms.string( "HLT" ),
    SeedThreshold = cms.int32( 1000 ),
    ClusterThreshold = cms.double( 4000.0 )
)
process.hltHISiPixelRecHits = cms.EDProducer( "SiPixelRecHitConverter",
    VerboseLevel = cms.untracked.int32( 0 ),
    src = cms.InputTag( "hltHISiPixelClusters" ),
    CPE = cms.string( "hltESPPixelCPEGeneric" )
)
process.hltHIPixelClusterVertices = cms.EDProducer( "HIPixelClusterVtxProducer",
    maxZ = cms.double( 20.05 ),
    zStep = cms.double( 0.1 ),
    minZ = cms.double( -20.0 ),
    pixelRecHits = cms.string( "hltHISiPixelRecHits" )
)
process.hltPixelTracksForHITrackTrigger = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( False ),
    FilterPSet = cms.PSet( 
      doVariablePtMin = cms.bool( True ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIProtoTrackFilter" ),
      ptMin = cms.double( 1.0 ),
      siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" ),
      tipMax = cms.double( 1.0 )
    ),
    passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" )
    ),
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
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
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
    )
)
process.hltPixelCandsForHITrackTrigger = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPixelTracksForHITrackTrigger" ),
    particleType = cms.string( "pi+" )
)
process.hltHISinglePixelTrackFilter = cms.EDFilter( "HLTPixlMBFilt",
    saveTags = cms.bool( False ),
    pixlTag = cms.InputTag( "hltPixelCandsForHITrackTrigger" ),
    MinTrks = cms.uint32( 1 ),
    MinPt = cms.double( 0.0 ),
    MinSep = cms.double( 1.0 )
)
process.hltL1sHIMinBiasBSC = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BscMinBiasThreshold1_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIMinBiasBSC = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIMinBiasBSCOR = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BscMinBiasOR_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIMinBiasBSCOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIMinBiasHF = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HcalHfCoincidencePm_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIMinBiasHF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIMinBiasHfOr = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HcalHfMmOrPpOrPm_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIMinBiasHfOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIMinBiasHfOrBSC = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HcalHfCoincPmORBscMinBiasThresh1_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIMinBiasHfOrBSC = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIMinBiasPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIMinBiasZDC = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZdcCaloPlus_ZdcCaloMinus_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIMinBiasZDCCalo = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIMinBiasZDCCaloPlusOrMinus = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZdcCaloPlus_BptxAND OR L1_ZdcCaloMinus_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIMinBiasZDCCaloPlusOrMinus = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIMinBiasZDCPixelSingleTrack = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZdcCaloPlus_ZdcCaloMinus_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIMinBiasZDCPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIMinBiasZDCPlusOrMinusPixelSingleTrack = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZdcCaloPlus_BptxAND OR L1_ZdcCaloMinus_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIMinBiasZDCPlusOrMinusPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIBptxXOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1BptxXORBscMinBiasOR = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxXOR_BscMinBiasOR" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1AlgoBptxXORBSCOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleMuOpenBptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "( L1_DoubleMuOpen_BptxAND OR L1_ETT2000 ) AND ( L1_HcalHfCoincPmORBscMinBiasThresh1_BptxAND_instance1 OR L1_NotBsc2_BscMinBiasOR OR L1_NotBsc2_HcalHfMmOrPpOrPm )" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1DoubleMuOpen = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoubleMuLevel1PathL1OpenFiltered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpenBptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreHIL1DoubleMu0HighQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoubleMuLevel1PathL1HighQFiltered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpenBptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32( 7, 6, 5, 0, 0, 0, 0, 0 ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltL1sL1SingleMu3BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "( L1_SingleMu3_BptxAND OR L1_ETT2000 ) AND ( L1_HcalHfCoincPmORBscMinBiasThresh1_BptxAND_instance1 OR L1_NotBsc2_BscMinBiasOR OR L1_NotBsc2_HcalHfMmOrPpOrPm )" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu3withBptxANDL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
    useStandardFEDid = cms.bool( True ),
    inputLabel = cms.InputTag( "rawDataCollector" ),
    dataType = cms.string( "DDU" ),
    fedbyType = cms.bool( False ),
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
    ),
    dqmOnly = cms.bool( False )
)
process.hltDt1DRecHits = cms.EDProducer( "DTRecHitProducer",
    debug = cms.untracked.bool( False ),
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
    dtDigiLabel = cms.InputTag( "hltMuonDTDigis" ),
    recAlgo = cms.string( "DTLinearDriftFromDBAlgo" )
)
process.hltDt4DSegments = cms.EDProducer( "DTRecSegment4DProducer",
    debug = cms.untracked.bool( False ),
    Reco4DAlgoName = cms.string( "DTCombinatorialPatternReco4D" ),
    recHits2DLabel = cms.InputTag( "dt2DSegments" ),
    recHits1DLabel = cms.InputTag( "hltDt1DRecHits" ),
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
process.hltMuonCSCDigis = cms.EDProducer( "CSCDCCUnpacker",
    PrintEventNumber = cms.untracked.bool( False ),
    UseSelectiveUnpacking = cms.bool( True ),
    UseExaminer = cms.bool( True ),
    ErrorMask = cms.uint32( 0x0 ),
    InputObjects = cms.InputTag( "rawDataCollector" ),
    UseFormatStatus = cms.bool( True ),
    ExaminerMask = cms.uint32( 0x1febf3f6 ),
    UnpackStatusDigis = cms.bool( False ),
    VisualFEDInspect = cms.untracked.bool( False ),
    FormatedEventDump = cms.untracked.bool( False ),
    Debug = cms.untracked.bool( False ),
    VisualFEDShort = cms.untracked.bool( False )
)
process.hltCsc2DRecHits = cms.EDProducer( "CSCRecHitDProducer",
    XTasymmetry_ME1b = cms.double( 0.0 ),
    XTasymmetry_ME1a = cms.double( 0.0 ),
    ConstSyst_ME1a = cms.double( 0.022 ),
    ConstSyst_ME1b = cms.double( 0.0070 ),
    XTasymmetry_ME41 = cms.double( 0.0 ),
    CSCStripxtalksOffset = cms.double( 0.03 ),
    CSCUseCalibrations = cms.bool( True ),
    CSCUseTimingCorrections = cms.bool( True ),
    CSCNoOfTimeBinsForDynamicPedestal = cms.int32( 2 ),
    XTasymmetry_ME22 = cms.double( 0.0 ),
    UseFivePoleFit = cms.bool( True ),
    XTasymmetry_ME21 = cms.double( 0.0 ),
    ConstSyst_ME21 = cms.double( 0.0 ),
    CSCDebug = cms.untracked.bool( False ),
    ConstSyst_ME22 = cms.double( 0.0 ),
    CSCUseGasGainCorrections = cms.bool( False ),
    XTasymmetry_ME31 = cms.double( 0.0 ),
    readBadChambers = cms.bool( True ),
    NoiseLevel_ME13 = cms.double( 8.0 ),
    NoiseLevel_ME12 = cms.double( 9.0 ),
    NoiseLevel_ME32 = cms.double( 9.0 ),
    NoiseLevel_ME31 = cms.double( 9.0 ),
    XTasymmetry_ME32 = cms.double( 0.0 ),
    ConstSyst_ME41 = cms.double( 0.0 ),
    CSCStripClusterSize = cms.untracked.int32( 3 ),
    CSCStripClusterChargeCut = cms.double( 25.0 ),
    CSCStripPeakThreshold = cms.double( 10.0 ),
    readBadChannels = cms.bool( True ),
    UseParabolaFit = cms.bool( False ),
    XTasymmetry_ME13 = cms.double( 0.0 ),
    XTasymmetry_ME12 = cms.double( 0.0 ),
    wireDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCWireDigi' ),
    ConstSyst_ME12 = cms.double( 0.0 ),
    ConstSyst_ME13 = cms.double( 0.0 ),
    ConstSyst_ME32 = cms.double( 0.0 ),
    ConstSyst_ME31 = cms.double( 0.0 ),
    UseAverageTime = cms.bool( False ),
    NoiseLevel_ME1a = cms.double( 7.0 ),
    NoiseLevel_ME1b = cms.double( 8.0 ),
    CSCWireClusterDeltaT = cms.int32( 1 ),
    CSCUseStaticPedestals = cms.bool( False ),
    stripDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCStripDigi' ),
    CSCstripWireDeltaTime = cms.int32( 8 ),
    NoiseLevel_ME21 = cms.double( 9.0 ),
    NoiseLevel_ME22 = cms.double( 9.0 ),
    NoiseLevel_ME41 = cms.double( 9.0 )
)
process.hltCscSegments = cms.EDProducer( "CSCSegmentProducer",
    inputObjects = cms.InputTag( "hltCsc2DRecHits" ),
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
    ),
    algo_type = cms.int32( 1 )
)
process.hltMuonRPCDigis = cms.EDProducer( "RPCUnpackingModule",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    doSynchro = cms.bool( False )
)
process.hltRpcRecHits = cms.EDProducer( "RPCRecHitProducer",
    recAlgoConfig = cms.PSet(  ),
    deadvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat" ),
    rpcDigiLabel = cms.InputTag( "hltMuonRPCDigis" ),
    maskvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat" ),
    recAlgo = cms.string( "RPCRecHitStandardAlgo" ),
    deadSource = cms.string( "File" ),
    maskSource = cms.string( "File" )
)
process.hltL2OfflineMuonSeeds = cms.EDProducer( "MuonSeedGenerator",
    SMB_21 = cms.vdouble( 1.043, -0.124, 0.0, 0.183, 0.0, 0.0 ),
    SMB_20 = cms.vdouble( 1.011, -0.052, 0.0, 0.188, 0.0, 0.0 ),
    SMB_22 = cms.vdouble( 1.474, -0.758, 0.0, 0.185, 0.0, 0.0 ),
    OL_2213 = cms.vdouble( 0.117, 0.0, 0.0, 0.044, 0.0, 0.0 ),
    SME_11 = cms.vdouble( 3.295, -1.527, 0.112, 0.378, 0.02, 0.0 ),
    SME_13 = cms.vdouble( -1.286, 1.711, 0.0, 0.356, 0.0, 0.0 ),
    SME_12 = cms.vdouble( 0.102, 0.599, 0.0, 0.38, 0.0, 0.0 ),
    DT_34_2_scale = cms.vdouble( -11.901897, 0.0 ),
    OL_1213_0_scale = cms.vdouble( -4.488158, 0.0 ),
    OL_1222_0_scale = cms.vdouble( -5.810449, 0.0 ),
    DT_13 = cms.vdouble( 0.315, 0.068, -0.127, 0.051, -0.0020, 0.0 ),
    DT_12 = cms.vdouble( 0.183, 0.054, -0.087, 0.028, 0.0020, 0.0 ),
    DT_14 = cms.vdouble( 0.359, 0.052, -0.107, 0.072, -0.0040, 0.0 ),
    CSC_13_3_scale = cms.vdouble( -1.701268, 0.0 ),
    DT_24_2_scale = cms.vdouble( -6.63094, 0.0 ),
    CSC_23 = cms.vdouble( -0.081, 0.113, -0.029, 0.015, 0.0080, 0.0 ),
    CSC_24 = cms.vdouble( 0.0040, 0.021, -0.0020, 0.053, 0.0, 0.0 ),
    OL_2222 = cms.vdouble( 0.107, 0.0, 0.0, 0.04, 0.0, 0.0 ),
    DT_14_2_scale = cms.vdouble( -4.808546, 0.0 ),
    SMB_10 = cms.vdouble( 1.387, -0.038, 0.0, 0.19, 0.0, 0.0 ),
    SMB_11 = cms.vdouble( 1.247, 0.72, -0.802, 0.229, -0.075, 0.0 ),
    SMB_12 = cms.vdouble( 2.128, -0.956, 0.0, 0.199, 0.0, 0.0 ),
    SME_21 = cms.vdouble( -0.529, 1.194, -0.358, 0.472, 0.086, 0.0 ),
    SME_22 = cms.vdouble( -1.207, 1.491, -0.251, 0.189, 0.243, 0.0 ),
    DT_13_2_scale = cms.vdouble( -4.257687, 0.0 ),
    CSC_34 = cms.vdouble( 0.062, -0.067, 0.019, 0.021, 0.0030, 0.0 ),
    SME_22_0_scale = cms.vdouble( -3.457901, 0.0 ),
    DT_24_1_scale = cms.vdouble( -7.490909, 0.0 ),
    OL_1232_0_scale = cms.vdouble( -5.964634, 0.0 ),
    DT_23_1_scale = cms.vdouble( -5.320346, 0.0 ),
    SME_13_0_scale = cms.vdouble( 0.104905, 0.0 ),
    SMB_22_0_scale = cms.vdouble( 1.346681, 0.0 ),
    CSC_12_1_scale = cms.vdouble( -6.434242, 0.0 ),
    DT_34 = cms.vdouble( 0.044, 0.0040, -0.013, 0.029, 0.0030, 0.0 ),
    SME_32 = cms.vdouble( -0.901, 1.333, -0.47, 0.41, 0.073, 0.0 ),
    SME_31 = cms.vdouble( -1.594, 1.482, -0.317, 0.487, 0.097, 0.0 ),
    CSC_13_2_scale = cms.vdouble( -6.077936, 0.0 ),
    crackEtas = cms.vdouble( 0.2, 1.6, 1.7 ),
    SME_11_0_scale = cms.vdouble( 1.325085, 0.0 ),
    SMB_20_0_scale = cms.vdouble( 1.486168, 0.0 ),
    DT_13_1_scale = cms.vdouble( -4.520923, 0.0 ),
    CSC_24_1_scale = cms.vdouble( -6.055701, 0.0 ),
    CSC_01_1_scale = cms.vdouble( -1.915329, 0.0 ),
    DT_23 = cms.vdouble( 0.13, 0.023, -0.057, 0.028, 0.0040, 0.0 ),
    DT_24 = cms.vdouble( 0.176, 0.014, -0.051, 0.051, 0.0030, 0.0 ),
    SMB_12_0_scale = cms.vdouble( 2.283221, 0.0 ),
    SMB_30_0_scale = cms.vdouble( -3.629838, 0.0 ),
    SME_42 = cms.vdouble( -0.0030, 0.0050, 0.0050, 0.608, 0.076, 0.0 ),
    SME_41 = cms.vdouble( -0.0030, 0.0050, 0.0050, 0.608, 0.076, 0.0 ),
    CSC_12_2_scale = cms.vdouble( -1.63622, 0.0 ),
    DT_34_1_scale = cms.vdouble( -13.783765, 0.0 ),
    CSC_34_1_scale = cms.vdouble( -11.520507, 0.0 ),
    OL_2213_0_scale = cms.vdouble( -7.239789, 0.0 ),
    SMB_32_0_scale = cms.vdouble( -3.054156, 0.0 ),
    CSC_12_3_scale = cms.vdouble( -1.63622, 0.0 ),
    SME_21_0_scale = cms.vdouble( -0.040862, 0.0 ),
    OL_1232 = cms.vdouble( 0.184, 0.0, 0.0, 0.066, 0.0, 0.0 ),
    DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
    SMB_10_0_scale = cms.vdouble( 2.448566, 0.0 ),
    EnableDTMeasurement = cms.bool( True ),
    CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
    CSC_23_2_scale = cms.vdouble( -6.079917, 0.0 ),
    scaleDT = cms.bool( True ),
    DT_12_2_scale = cms.vdouble( -3.518165, 0.0 ),
    OL_1222 = cms.vdouble( 0.848, -0.591, 0.0, 0.062, 0.0, 0.0 ),
    CSC_23_1_scale = cms.vdouble( -19.084285, 0.0 ),
    OL_1213 = cms.vdouble( 0.96, -0.737, 0.0, 0.052, 0.0, 0.0 ),
    CSC_02 = cms.vdouble( 0.612, -0.207, 0.0, 0.067, -0.0010, 0.0 ),
    CSC_03 = cms.vdouble( 0.787, -0.338, 0.029, 0.101, -0.0080, 0.0 ),
    CSC_01 = cms.vdouble( 0.166, 0.0, 0.0, 0.031, 0.0, 0.0 ),
    SMB_32 = cms.vdouble( 0.67, -0.327, 0.0, 0.22, 0.0, 0.0 ),
    SMB_30 = cms.vdouble( 0.505, -0.022, 0.0, 0.215, 0.0, 0.0 ),
    SMB_31 = cms.vdouble( 0.549, -0.145, 0.0, 0.207, 0.0, 0.0 ),
    crackWindow = cms.double( 0.04 ),
    CSC_14_3_scale = cms.vdouble( -1.969563, 0.0 ),
    SMB_31_0_scale = cms.vdouble( -3.323768, 0.0 ),
    DT_12_1_scale = cms.vdouble( -3.692398, 0.0 ),
    SMB_21_0_scale = cms.vdouble( 1.58384, 0.0 ),
    DT_23_2_scale = cms.vdouble( -5.117625, 0.0 ),
    SME_12_0_scale = cms.vdouble( 2.279181, 0.0 ),
    DT_14_1_scale = cms.vdouble( -5.644816, 0.0 ),
    beamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    SMB_11_0_scale = cms.vdouble( 2.56363, 0.0 ),
    EnableCSCMeasurement = cms.bool( True ),
    CSC_14 = cms.vdouble( 0.606, -0.181, -0.0020, 0.111, -0.0030, 0.0 ),
    OL_2222_0_scale = cms.vdouble( -7.667231, 0.0 ),
    CSC_13 = cms.vdouble( 0.901, -1.302, 0.533, 0.045, 0.0050, 0.0 ),
    CSC_12 = cms.vdouble( -0.161, 0.254, -0.047, 0.042, -0.0070, 0.0 )
)
process.hltL2MuonSeeds = cms.EDProducer( "L2MuonSeedGenerator",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    InputObjects = cms.InputTag( "hltL1extraParticles" ),
    L1MaxEta = cms.double( 2.5 ),
    OfflineSeedLabel = cms.untracked.InputTag( "hltL2OfflineMuonSeeds" ),
    L1MinPt = cms.double( 0.0 ),
    L1MinQuality = cms.uint32( 1 ),
    GMTReadoutCollection = cms.InputTag( "hltGtDigis" ),
    UseOfflineSeed = cms.untracked.bool( True ),
    Propagator = cms.string( "SteppingHelixPropagatorAny" )
)
process.hltL2Muons = cms.EDProducer( "L2MuonProducer",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny',
        'hltESPFastSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    InputObjects = cms.InputTag( "hltL2MuonSeeds" ),
    SeedTransformerParameters = cms.PSet( 
      Fitter = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      NMinRecHits = cms.uint32( 2 ),
      UseSubRecHits = cms.bool( False ),
      Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      RescaleError = cms.double( 100.0 )
    ),
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
    DoSeedRefit = cms.bool( False ),
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
process.hltL2MuonCandidates = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltHIL2Mu3L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3withBptxANDL1Filtered" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL2Mu3NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2Mu3NHitL2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3withBptxANDL1Filtered" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 3.0 ),
    MinNhits = cms.vint32( 1 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL2Mu7 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2Mu7L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3withBptxANDL1Filtered" ),
    MinPt = cms.double( 7.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL2Mu15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2Mu15L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3withBptxANDL1Filtered" ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL2DoubleMu0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2DoubleMu0L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMuLevel1PathL1OpenFiltered" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 3.0 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL2DoubleMu0NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2DoubleMu0L2N1HitsFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMuLevel1PathL1OpenFiltered" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 3.0 ),
    MinNhits = cms.vint32( 1 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL2DoubleMu0L1HighQL2NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIL2DoubleMu3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2DoubleMu3L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMuLevel1PathL1OpenFiltered" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 2 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 3.0 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL3Mu3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSiStripExcludedFEDListProducer = cms.EDProducer( "SiStripExcludedFEDListProducer",
    ProductLabel = cms.InputTag( "rawDataCollector" )
)
process.hltHISiStripRawToClustersFacility = cms.EDProducer( "SiStripRawToClusters",
    ProductLabel = cms.InputTag( "rawDataCollector" ),
    DoAPVEmulatorCheck = cms.bool( False ),
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
    ),
    Clusterizer = cms.PSet( 
      ChannelThreshold = cms.double( 2.0 ),
      MaxSequentialBad = cms.uint32( 1 ),
      MaxSequentialHoles = cms.uint32( 0 ),
      Algorithm = cms.string( "ThreeThresholdAlgorithm" ),
      MaxAdjacentBad = cms.uint32( 0 ),
      QualityLabel = cms.string( "" ),
      SeedThreshold = cms.double( 3.0 ),
      ClusterThreshold = cms.double( 5.0 ),
      setDetId = cms.bool( True ),
      RemoveApvShots = cms.bool( True )
    )
)
process.hltHISiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltHISiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "hltHIAllESPMeasurementTracker" )
)
process.hltL3TrajSeedOIState = cms.EDProducer( "TSGFromL2Muon",
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
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSteppingHelixPropagatorOpposite',
        'hltESPSteppingHelixPropagatorAlong' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    MuonTrackingRegionBuilder = cms.PSet(  ),
    PCut = cms.double( 2.5 ),
    TrackerSeedCleaner = cms.PSet(  ),
    PtCut = cms.double( 1.0 )
)
process.hltHIL3TrackCandidateFromL2OIState = cms.EDProducer( "CkfTrajectoryMaker",
    src = cms.InputTag( "hltL3TrajSeedOIState" ),
    reverseTrajectories = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    trackCandidateAlso = cms.bool( True ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltHIAllESPMuonCkfTrajectoryBuilder" ),
    maxNSeeds = cms.uint32( 100000 )
)
process.hltHIL3TkTracksFromL2OIState = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIL3TrackCandidateFromL2OIState" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    Propagator = cms.string( "PropagatorWithMaterial" )
)
process.hltHIL3MuonsOIState = cms.EDProducer( "L3MuonProducer",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
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
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltHIL3TrajSeedOIHit = cms.EDProducer( "TSGFromL2Muon",
    TkSeedGenerator = cms.PSet( 
      PSetNames = cms.vstring( 'skipTSG',
        'iterativeTSG' ),
      L3TkCollectionA = cms.InputTag( "hltHIL3MuonsOIState" ),
      iterativeTSG = cms.PSet( 
        ErrorRescaling = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "unused" ),
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
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'PropagatorWithMaterial',
        'hltESPSmartPropagatorAnyOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    MuonTrackingRegionBuilder = cms.PSet(  ),
    PCut = cms.double( 2.5 ),
    TrackerSeedCleaner = cms.PSet( 
      cleanerFromSharedHits = cms.bool( True ),
      ptCleaner = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      directionCleaner = cms.bool( True )
    ),
    PtCut = cms.double( 1.0 )
)
process.hltHIL3TrackCandidateFromL2OIHit = cms.EDProducer( "CkfTrajectoryMaker",
    src = cms.InputTag( "hltHIL3TrajSeedOIHit" ),
    reverseTrajectories = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    trackCandidateAlso = cms.bool( True ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltHIAllESPMuonCkfTrajectoryBuilder" ),
    maxNSeeds = cms.uint32( 100000 )
)
process.hltHIL3TkTracksFromL2OIHit = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIL3TrackCandidateFromL2OIHit" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    Propagator = cms.string( "PropagatorWithMaterial" )
)
process.hltHIL3MuonsOIHit = cms.EDProducer( "L3MuonProducer",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
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
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltHIL3TkFromL2OICombination = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit' )
)
process.hltHIL3TrajSeedIOHit = cms.EDProducer( "TSGFromL2Muon",
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
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'PropagatorWithMaterial' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
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
    PCut = cms.double( 2.5 ),
    TrackerSeedCleaner = cms.PSet( 
      cleanerFromSharedHits = cms.bool( True ),
      ptCleaner = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      directionCleaner = cms.bool( True )
    ),
    PtCut = cms.double( 1.0 )
)
process.hltHIL3TrackCandidateFromL2IOHit = cms.EDProducer( "CkfTrajectoryMaker",
    src = cms.InputTag( "hltHIL3TrajSeedIOHit" ),
    reverseTrajectories = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    trackCandidateAlso = cms.bool( True ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltHIAllESPMuonCkfTrajectoryBuilder" ),
    maxNSeeds = cms.uint32( 100000 )
)
process.hltHIL3TkTracksFromL2IOHit = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIL3TrackCandidateFromL2IOHit" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),
    Propagator = cms.string( "PropagatorWithMaterial" )
)
process.hltHIAllL3MuonsIOHit = cms.EDProducer( "L3MuonProducer",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
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
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltHIL3TrajectorySeed = cms.EDProducer( "L3MuonTrajectorySeedCombiner",
    labels = cms.VInputTag( 'hltHIL3TrajSeedIOHit','hltL3TrajSeedOIState','hltHIL3TrajSeedOIHit' )
)
process.hltHIL3TrackCandidateFromL2 = cms.EDProducer( "L3TrackCandCombiner",
    labels = cms.VInputTag( 'hltHIL3TrackCandidateFromL2IOHit','hltHIL3TrackCandidateFromL2OIHit','hltHIL3TrackCandidateFromL2OIState' )
)
process.hltHIL3TkTracksFromL2 = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltHIL3TkTracksFromL2IOHit','hltHIL3TkTracksFromL2OIHit','hltHIL3TkTracksFromL2OIState' )
)
process.hltHIL3MuonsLinksCombination = cms.EDProducer( "L3TrackLinksCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit','hltHIAllL3MuonsIOHit' )
)
process.hltHIL3Muons = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit','hltHIAllL3MuonsIOHit' )
)
process.hltHIL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputLinksObjects = cms.InputTag( "hltHIL3MuonsLinksCombination" ),
    InputObjects = cms.InputTag( "hltHIL3Muons" ),
    MuonPtOption = cms.string( "Global" )
)
process.hltHISingleMu3L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 9999.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL2Mu3L2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 9999.0 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinPt = cms.double( 3.0 )
)
process.hltPreHIL3DoubleMuOpen = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDimuonL2PreFiltered0 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMuLevel1PathL1OpenFiltered" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltHIDimuonL3FilteredOpen = cms.EDFilter( "HLTMuonDimuonL3Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( 0 ),
    MaxPtMin = cms.vdouble( 1.0E125 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonL2PreFiltered0" ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxPtPair = cms.vdouble( 1.0E125 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtMin = cms.vdouble( 0.0 ),
    MaxInvMass = cms.vdouble( 300.0 ),
    MinPtMax = cms.vdouble( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.vdouble( 0.0 ),
    MaxDr = cms.double( 20.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MinNhits = cms.int32( 0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    CutCowboys = cms.bool( False ),
    MinInvMass = cms.vdouble( 0.0 )
)
process.hltPreHIL3DoubleMuOpenMgt2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDimuonL3FilteredMgt2 = cms.EDFilter( "HLTMuonDimuonL3Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( 0 ),
    MaxPtMin = cms.vdouble( 1.0E125 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonL2PreFiltered0" ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxPtPair = cms.vdouble( 1.0E125 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtMin = cms.vdouble( 0.0 ),
    MaxInvMass = cms.vdouble( 300.0 ),
    MinPtMax = cms.vdouble( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.vdouble( 0.0 ),
    MaxDr = cms.double( 20.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MinNhits = cms.int32( 0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    CutCowboys = cms.bool( False ),
    MinInvMass = cms.vdouble( 2.0 )
)
process.hltPreHIL3DoubleMuOpenMgt2SS = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDimuonL3FilteredMgt2SS = cms.EDFilter( "HLTMuonDimuonL3Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( 1 ),
    MaxPtMin = cms.vdouble( 1.0E125 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonL2PreFiltered0" ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxPtPair = cms.vdouble( 1.0E125 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtMin = cms.vdouble( 0.0 ),
    MaxInvMass = cms.vdouble( 300.0 ),
    MinPtMax = cms.vdouble( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.vdouble( 0.0 ),
    MaxDr = cms.double( 20.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MinNhits = cms.int32( 0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    CutCowboys = cms.bool( False ),
    MinInvMass = cms.vdouble( 2.0 )
)
process.hltPreHIL3DoubleMuOpenMgt2OS = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDimuonL3FilteredMg2OS = cms.EDFilter( "HLTMuonDimuonL3Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( -1 ),
    MaxPtMin = cms.vdouble( 1.0E125 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonL2PreFiltered0" ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxPtPair = cms.vdouble( 1.0E125 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtMin = cms.vdouble( 0.0 ),
    MaxInvMass = cms.vdouble( 300.0 ),
    MinPtMax = cms.vdouble( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.vdouble( 0.0 ),
    MaxDr = cms.double( 20.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MinNhits = cms.int32( 0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    CutCowboys = cms.bool( False ),
    MinInvMass = cms.vdouble( 2.0 )
)
process.hltPreHIL3DoubleMuOpenMgt2OSNoCowboy = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDimuonL3FilteredMg2OSnoCowboy = cms.EDFilter( "HLTMuonDimuonL3Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( -1 ),
    MaxPtMin = cms.vdouble( 1.0E125 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonL2PreFiltered0" ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxPtPair = cms.vdouble( 1.0E125 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtMin = cms.vdouble( 0.0 ),
    MaxInvMass = cms.vdouble( 300.0 ),
    MinPtMax = cms.vdouble( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.vdouble( 0.0 ),
    MaxDr = cms.double( 20.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MinNhits = cms.int32( 0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    CutCowboys = cms.bool( True ),
    MinInvMass = cms.vdouble( 2.0 )
)
process.hltL1sL1SingleEG5BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5_BptxAND OR L1_ETT2000" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHISinglePhoton15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltIslandBasicClustersHI = cms.EDProducer( "IslandClusterProducer",
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClustersHI" ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    posCalcParameters = cms.PSet( 
      T0_barl = cms.double( 7.4 ),
      LogWeighted = cms.bool( True ),
      T0_endc = cms.double( 3.1 ),
      T0_endcPresh = cms.double( 1.2 ),
      W0 = cms.double( 4.2 ),
      X0 = cms.double( 0.89 )
    ),
    barrelShapeAssociation = cms.string( "islandBarrelShapeAssoc" ),
    endcapShapeAssociation = cms.string( "islandEndcapShapeAssoc" ),
    endcapHitProducer = cms.string( "hltEcalRecHitAll" ),
    clustershapecollectionEB = cms.string( "islandBarrelShape" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    barrelHitProducer = cms.string( "hltEcalRecHitAll" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    clustershapecollectionEE = cms.string( "islandEndcapShape" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClustersHI" ),
    VerbosityLevel = cms.string( "ERROR" )
)
process.hltHiIslandSuperClustersHI = cms.EDProducer( "HiSuperClusterProducer",
    barrelSuperclusterCollection = cms.string( "islandBarrelSuperClustersHI" ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClustersHI" ),
    endcapClusterProducer = cms.string( "hltIslandBasicClustersHI" ),
    barrelPhiSearchRoad = cms.double( 0.8 ),
    endcapPhiSearchRoad = cms.double( 0.6 ),
    endcapBCEnergyThreshold = cms.double( 0.0 ),
    VerbosityLevel = cms.string( "ERROR" ),
    seedTransverseEnergyThreshold = cms.double( 1.0 ),
    barrelEtaSearchRoad = cms.double( 0.07 ),
    endcapSuperclusterCollection = cms.string( "islandEndcapSuperClustersHI" ),
    barrelBCEnergyThreshold = cms.double( 0.0 ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( True ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClustersHI" ),
    barrelClusterProducer = cms.string( "hltIslandBasicClustersHI" )
)
process.hltHiCorrectedIslandBarrelSuperClustersHI = cms.EDProducer( "HiEgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string( "" ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    superClusterAlgo = cms.string( "Island" ),
    etThresh = cms.double( 0.0 ),
    rawSuperClusterProducer = cms.InputTag( 'hltHiIslandSuperClustersHI','islandBarrelSuperClustersHI' ),
    applyEnergyCorrection = cms.bool( True ),
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
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' )
)
process.hltHiCorrectedIslandEndcapSuperClustersHI = cms.EDProducer( "HiEgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string( "" ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    superClusterAlgo = cms.string( "Island" ),
    etThresh = cms.double( 0.0 ),
    rawSuperClusterProducer = cms.InputTag( 'hltHiIslandSuperClustersHI','islandEndcapSuperClustersHI' ),
    applyEnergyCorrection = cms.bool( True ),
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
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' )
)
process.hltCleanedHiCorrectedIslandBarrelSuperClustersHI = cms.EDProducer( "HiSpikeCleaner",
    originalSuperClusterProducer = cms.InputTag( "hltHiCorrectedIslandBarrelSuperClustersHI" ),
    recHitProducerEndcap = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    TimingCut = cms.untracked.double( 9999999.0 ),
    swissCutThr = cms.untracked.double( 0.95 ),
    recHitProducerBarrel = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    etCut = cms.double( 8.0 ),
    outputColl = cms.string( "" )
)
process.hltRecoHIEcalWithCleaningCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scIslandEndcapProducer = cms.InputTag( "hltHiCorrectedIslandEndcapSuperClustersHI" ),
    scHybridBarrelProducer = cms.InputTag( "hltCleanedHiCorrectedIslandBarrelSuperClustersHI" ),
    recoEcalCandidateCollection = cms.string( "" )
)
process.hltHIPhoton15 = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton20 = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton30 = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 30.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton40 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton40 = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltL1sL1DoubleEG5BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG5_BptxAND OR L1_ETT2000" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPhoton10Photon15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoublePhoton1015Filter1 = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 10.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltHIDoublePhoton1015Filter2 = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHIPhoton15Photon20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoublePhoton1520Filter1 = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltHIDoublePhoton1520Filter2 = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHIDoublePhoton10 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoublePhoton10 = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 10.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHIDoublePhoton15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoublePhoton15 = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHIDoublePhoton20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoublePhoton20 = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltL1sL1SingleJet36BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet36_BptxAND OR L1_ETT2000" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIJet55 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltIterativeCone5PileupSubtractionCaloJets = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( -9.0 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "CaloJet" ),
    minSeed = cms.uint32( 0 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "IterativeCone" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltTowerMakerForAll" ),
    inputEtMin = cms.double( 0.3 ),
    puPtMin = cms.double( 10.0 ),
    srcPVs = cms.InputTag( "offlinePrimaryVertices" ),
    jetPtMin = cms.double( 10.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( True ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "MultipleAlgoIterator" ),
    MinVtxNdof = cms.int32( 5 ),
    MaxVtxZ = cms.double( 15.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    DzTrVtxMax = cms.double( 0.0 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.0 )
)
process.hltHICaloJetIDPassed = cms.EDProducer( "HLTCaloJetIDProducer",
    min_N90 = cms.int32( -2 ),
    min_N90hits = cms.int32( -2 ),
    min_EMF = cms.double( 1.0E-6 ),
    jetsInput = cms.InputTag( "hltIterativeCone5PileupSubtractionCaloJets" ),
    JetIDParams = cms.PSet( 
      useRecHits = cms.bool( True ),
      hbheRecHitsColl = cms.InputTag( "hltHbhereco" ),
      hoRecHitsColl = cms.InputTag( "hltHoreco" ),
      hfRecHitsColl = cms.InputTag( "hltHfreco" ),
      ebRecHitsColl = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
      eeRecHitsColl = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' )
    ),
    max_EMF = cms.double( 999.0 )
)
process.hltHICaloJetCorrected = cms.EDProducer( "CaloJetCorrectionProducer",
    src = cms.InputTag( "hltHICaloJetIDPassed" ),
    correctors = cms.vstring( 'hltESPAK5CaloL2L3' )
)
process.hltHI1jet55 = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 55.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltHICaloJetCorrected" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHIJet65 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHI1jet65 = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 65.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltHICaloJetCorrected" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleJet52BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet52_BptxAND OR L1_ETT2000" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIJet80 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHI1jet80 = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 80.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltHICaloJetCorrected" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleJet68BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet52_BptxAND OR L1_ETT2000" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIJet95 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHI1jet95 = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 95.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltHICaloJetCorrected" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHIDiJet55 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDiJet55and55 = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 55.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 5.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltHICaloJetCorrected" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHIJet65Jet55 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleJet20CentralNotBptx = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet20_Central_NotBptxOR_NotMuBeamHalo" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIJetE30NoBPTX = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltStoppedHSCPTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
    EBSumThreshold = cms.double( 0.2 ),
    MomHBDepth = cms.double( 0.2 ),
    UseEtEBTreshold = cms.bool( False ),
    hfInput = cms.InputTag( "" ),
    AllowMissingInputs = cms.bool( True ),
    MomEEDepth = cms.double( 0.0 ),
    EESumThreshold = cms.double( 0.45 ),
    HBGrid = cms.vdouble(  ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    HBThreshold = cms.double( 0.7 ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  ),
    UseEcalRecoveredHits = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHEDepth = cms.double( 0.4 ),
    HcalThreshold = cms.double( -1000.0 ),
    HF2Weights = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    UseSymEBTreshold = cms.bool( False ),
    EEWeights = cms.vdouble(  ),
    EEWeight = cms.double( 1.0 ),
    UseHO = cms.bool( False ),
    HBWeights = cms.vdouble(  ),
    HF1Weight = cms.double( 1.0 ),
    HF2Grid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    EBWeight = cms.double( 1.0 ),
    HF1Grid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    HOWeight = cms.double( 1.0E-99 ),
    HESWeight = cms.double( 1.0 ),
    HESThreshold = cms.double( 0.8 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    HF2Weight = cms.double( 1.0 ),
    HF2Threshold = cms.double( 0.85 ),
    HcalAcceptSeverityLevel = cms.uint32( 9 ),
    EEThreshold = cms.double( 0.3 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HF1Weights = cms.vdouble(  ),
    hoInput = cms.InputTag( "" ),
    HF1Threshold = cms.double( 0.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HESGrid = cms.vdouble(  ),
    EcutTower = cms.double( -1000.0 ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    HESWeights = cms.vdouble(  ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kTime',
      'kWeird',
      'kBad' ),
    HEDWeight = cms.double( 1.0 ),
    UseSymEETreshold = cms.bool( False ),
    HEDThreshold = cms.double( 0.8 ),
    EBThreshold = cms.double( 0.07 ),
    UseRejectedHitsOnly = cms.bool( False ),
    UseHcalRecoveredHits = cms.bool( False ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HOThreshold0 = cms.double( 3.5 ),
    ecalInputs = cms.VInputTag(  ),
    UseRejectedRecoveredHcalHits = cms.bool( False ),
    MomEBDepth = cms.double( 0.3 ),
    HBWeight = cms.double( 1.0 ),
    HOGrid = cms.vdouble(  ),
    EBGrid = cms.vdouble(  )
)
process.hltStoppedHSCPIterativeCone5CaloJets = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 5 ),
    doAreaFastjet = cms.bool( False ),
    voronoiRfact = cms.double( -9.0 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "CaloJet" ),
    minSeed = cms.uint32( 0 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "IterativeCone" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.5 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltStoppedHSCPTowerMakerForAll" ),
    inputEtMin = cms.double( 0.3 ),
    puPtMin = cms.double( 10.0 ),
    srcPVs = cms.InputTag( "offlinePrimaryVertices" ),
    jetPtMin = cms.double( 1.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( False ),
    inputEMin = cms.double( 0.0 ),
    subtractorName = cms.string( "" ),
    MinVtxNdof = cms.int32( 5 ),
    MaxVtxZ = cms.double( 15.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    DzTrVtxMax = cms.double( 0.0 ),
    sumRecHits = cms.bool( False ),
    DxyTrVtxMax = cms.double( 0.0 )
)
process.hltStoppedHSCP1CaloJetEnergy30 = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( -1.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltStoppedHSCPIterativeCone5CaloJets" ),
    MinE = cms.double( 30.0 ),
    triggerType = cms.int32( 85 )
)
process.hltBPTXAntiCoincidence = cms.EDFilter( "HLTLevel1Activity",
    technicalBits = cms.uint64( 0x8 ),
    ignoreL1Mask = cms.bool( True ),
    invert = cms.bool( True ),
    physicsLoBits = cms.uint64( 0x0 ),
    physicsHiBits = cms.uint64( 0x0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    daqPartitions = cms.uint32( 1 ),
    bunchCrossings = cms.vint32( 0, 1, -1 )
)
process.hltL1sL1SingleJet32CentralNotBptx = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet32_Central_NotBptxOR_NotMuBeamHalo" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltL1BeamHaloAntiCoincidence3BX = cms.EDFilter( "HLTLevel1Activity",
    technicalBits = cms.uint64( 0x0 ),
    ignoreL1Mask = cms.bool( True ),
    invert = cms.bool( True ),
    physicsLoBits = cms.uint64( 0x0 ),
    physicsHiBits = cms.uint64( 0x8000000000000000 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    daqPartitions = cms.uint32( 1 ),
    bunchCrossings = cms.vint32( 0, 1, -1 )
)
process.hltPreHIJetE50NoBPTX3BXNoHalo = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltStoppedHSCPHpdFilter = cms.EDFilter( "HLTHPDFilter",
    rbxSpikeEnergy = cms.double( 50.0 ),
    energy = cms.double( -99.0 ),
    inputTag = cms.InputTag( "hltHbhereco" ),
    hpdSpikeIsolationEnergy = cms.double( 1.0 ),
    hpdSpikeEnergy = cms.double( 10.0 ),
    rbxSpikeUnbalance = cms.double( 0.2 )
)
process.hltStoppedHSCP1CaloJetEnergy50 = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( -1.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.0 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltStoppedHSCPIterativeCone5CaloJets" ),
    MinE = cms.double( 50.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1GlobalDecision = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1GlobalDecision" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIActivityHFCoincidence3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHcalSimpleRecHitFilterCoincidence = cms.EDFilter( "HLTHcalSimpleRecHitFilter",
    saveTags = cms.bool( False ),
    minNHitsNeg = cms.int32( 1 ),
    minNHitsPos = cms.int32( 1 ),
    threshold = cms.double( 3.0 ),
    doCoincidence = cms.bool( True ),
    maskedChannels = cms.vint32(  ),
    HFRecHitCollection = cms.InputTag( "hltHfreco" )
)
process.hltPreHIActivityHFSingle3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHcalSimpleRecHitFilter = cms.EDFilter( "HLTHcalSimpleRecHitFilter",
    saveTags = cms.bool( False ),
    minNHitsNeg = cms.int32( 1 ),
    minNHitsPos = cms.int32( 1 ),
    threshold = cms.double( 3.0 ),
    doCoincidence = cms.bool( False ),
    maskedChannels = cms.vint32(  ),
    HFRecHitCollection = cms.InputTag( "hltHfreco" )
)
process.hltPreHIClusterVertexCompatibility = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPixelClusterShapeFilter = cms.EDFilter( "HLTPixelClusterShapeFilter",
    maxZ = cms.double( 20.05 ),
    saveTags = cms.bool( False ),
    clusterTrunc = cms.double( 2.0 ),
    inputTag = cms.InputTag( "hltHISiPixelRecHits" ),
    zStep = cms.double( 0.2 ),
    minZ = cms.double( -20.0 ),
    nhitsTrunc = cms.int32( 150 ),
    clusterPars = cms.vdouble(  )
)
process.hltPreHICentralityVeto = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPixelActivityFilterCentralityVeto = cms.EDFilter( "HLTPixelActivityFilter",
    maxClusters = cms.uint32( 1000 ),
    saveTags = cms.bool( False ),
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 3 )
)
process.hltBPTXCoincidence = cms.EDFilter( "HLTLevel1Activity",
    technicalBits = cms.uint64( 0x1 ),
    ignoreL1Mask = cms.bool( True ),
    invert = cms.bool( False ),
    physicsLoBits = cms.uint64( 0x1 ),
    physicsHiBits = cms.uint64( 0x40000 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    daqPartitions = cms.uint32( 1 ),
    bunchCrossings = cms.vint32( 0, -1, 1 )
)
process.hltL1sETT100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETT100_BptxAND OR L1_ETT2000" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIFullTrack12L1Central = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHICaloTowerFilter4 = cms.EDFilter( "HLTCaloTowerFilter",
    MaxEta = cms.double( 2.4 ),
    saveTags = cms.bool( False ),
    inputTag = cms.InputTag( "hltTowerMakerForAll" ),
    MinPt = cms.double( 4.0 ),
    MinN = cms.uint32( 1 )
)
process.hltHIPixelClusterVerticesForHITrackTrigger = cms.EDProducer( "HIPixelClusterVtxProducer",
    maxZ = cms.double( 15.0 ),
    zStep = cms.double( 1.0 ),
    minZ = cms.double( -15.0 ),
    pixelRecHits = cms.string( "hltHISiPixelRecHits" )
)
process.hltHIPixel3ProtoTracks = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( False ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIProtoTrackFilter" ),
      ptMin = cms.double( 1.0 ),
      tipMax = cms.double( 1.0 ),
      doVariablePtMin = cms.bool( True ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" )
    ),
    passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
    ),
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
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
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
    )
)
process.hltHIPixelMedianVertex = cms.EDProducer( "HIPixelMedianVtxProducer",
    PeakFindThreshold = cms.uint32( 100 ),
    PeakFindMaxZ = cms.double( 30.0 ),
    FitThreshold = cms.int32( 5 ),
    TrackCollection = cms.InputTag( "hltHIPixel3ProtoTracks" ),
    PtMin = cms.double( 0.075 ),
    PeakFindBinsPerCm = cms.int32( 10 ),
    FitMaxZ = cms.double( 0.1 ),
    FitBinsPerCm = cms.int32( 500 )
)
process.hltHISelectedProtoTracks = cms.EDFilter( "HIProtoTrackSelection",
    src = cms.InputTag( "hltHIPixel3ProtoTracks" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    maxD0Significance = cms.double( 5.0 ),
    minZCut = cms.double( 0.2 ),
    VertexCollection = cms.InputTag( "hltHIPixelMedianVertex" ),
    ptMin = cms.double( 0.0 ),
    nSigmaZ = cms.double( 5.0 )
)
process.hltHIPixelAdaptiveVertex = cms.EDProducer( "PrimaryVertexProducer",
    vertexCollections = cms.VPSet( 
      cms.PSet(  maxDistanceToBeam = cms.double( 2.0 ),
        useBeamConstraint = cms.bool( False ),
        minNdof = cms.double( 0.0 ),
        algorithm = cms.string( "AdaptiveVertexFitter" ),
        label = cms.string( "" )
      )
    ),
    verbose = cms.untracked.bool( False ),
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
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    TrackLabel = cms.InputTag( "hltHISelectedProtoTracks" ),
    TkClusParameters = cms.PSet( 
      algorithm = cms.string( "gap" ),
      TkGapClusParameters = cms.PSet(  zSeparation = cms.double( 1.0 ) )
    )
)
process.hltHIBestAdaptiveVertex = cms.EDFilter( "HIBestVertexSelection",
    maxNumber = cms.uint32( 1 ),
    src = cms.InputTag( "hltHIPixelAdaptiveVertex" )
)
process.hltHISelectedVertex = cms.EDProducer( "HIBestVertexProducer",
    adaptiveVertexCollection = cms.InputTag( "hltHIBestAdaptiveVertex" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    medianVertexCollection = cms.InputTag( "hltHIPixelMedianVertex" )
)
process.hltHIPixel3PrimTracks = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( True ),
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
    passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
    ),
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
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "TrackCleaner" ) ),
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
    )
)
process.hltHIPixelTrackCandsForHITrackTrigger = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIPixel3PrimTracks" ),
    particleType = cms.string( "pi+" )
)
process.hltHISinglePixelTrackFilter10 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 10.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltHIPixelTrackCandsForHITrackTrigger" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 0.2 )
)
process.hltHISiStripClustersNonRegional = cms.EDProducer( "SiStripClusterizer",
    DigiProducersList = cms.VInputTag( 'hltSiStripZeroSuppression:VirginRaw','hltSiStripZeroSuppression:ProcessedRaw','hltSiStripZeroSuppression:ScopeMode' ),
    Clusterizer = cms.PSet( 
      ChannelThreshold = cms.double( 2.0 ),
      MaxSequentialBad = cms.uint32( 1 ),
      Algorithm = cms.string( "ThreeThresholdAlgorithm" ),
      MaxSequentialHoles = cms.uint32( 0 ),
      MaxAdjacentBad = cms.uint32( 0 ),
      QualityLabel = cms.string( "" ),
      SeedThreshold = cms.double( 3.0 ),
      ClusterThreshold = cms.double( 5.0 ),
      RemoveApvShots = cms.bool( True )
    )
)
process.hltHIPixelTrackSeeds = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    useEventsWithNoVertex = cms.bool( True ),
    originHalfLength = cms.double( 1.0E9 ),
    useProtoTrackKinematics = cms.bool( False ),
    InputVertexCollection = cms.InputTag( "" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    InputCollection = cms.InputTag( "hltHIPixel3PrimTracks" ),
    originRadius = cms.double( 1.0E9 )
)
process.hltHIPrimTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltHIPixelTrackSeeds" ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialForHI" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOppositeForHI" ),
      numberMeasurementsForFit = cms.int32( 4 )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedSeeds" ),
    cleanTrajectoryAfterInOut = cms.bool( True ),
    useHitsSplitting = cms.bool( True ),
    RedundantSeedCleaner = cms.string( "none" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 100000 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "hltESPCkfTrajectoryBuilderForHI" )
)
process.hltHIGlobalPrimTracks = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIPrimTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    Fitter = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
    useHitsSplitting = cms.bool( True ),
    MeasurementTracker = cms.string( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
    AlgorithmName = cms.string( "PropagatorWithMaterialOppositeForHI" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltHIGoodLooseTracks = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "loose" ),
    minNumberLayers = cms.uint32( 6 ),
    chi2n_par = cms.double( 9999.0 ),
    useVtxError = cms.bool( True ),
    nSigmaZ = cms.double( 4.0 ),
    dz_par2 = cms.vdouble( 5.0, 0.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 9999.0, 0.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 0.2 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 999 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 0.055 ),
    copyExtras = cms.untracked.bool( False ),
    max_z0NoPV = cms.double( 15.0 ),
    vertexCut = cms.string( "" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 11 ),
    src = cms.InputTag( "hltHIGlobalPrimTracks" ),
    chi2n_no1Dmod_par = cms.double( 0.2 ),
    vertices = cms.InputTag( "hltHISelectedVertex" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 5.0, 0.0 ),
    d0_par1 = cms.vdouble( 9999.0, 0.0 ),
    res_par = cms.vdouble( 99999.0, 99999.0 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltHIFullTrackCandsForHITrackTrigger = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIGoodLooseTracks" ),
    particleType = cms.string( "pi+" )
)
process.hltHISingleFullTrackFilter12 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 12.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 0.2 )
)
process.hltL1sJet16CentralNotETT140BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet16_Central_NotETT140_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIFullTrack12L1Peripheral = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIFullTrack14L1Central = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHISingleFullTrackFilter14 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 14.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 0.2 )
)
process.hltPreHIFullTrack14L1Peripheral = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIFullTrack20L1Central = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHISingleFullTrackFilter20 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 20.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 0.2 )
)
process.hltPreHIFullTrack20L1Peripheral = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIFullTrack25L1Central = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHISingleFullTrackFilter25 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 25.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 2.4 ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 0.2 )
)
process.hltPreHIFullTrack25L1Peripheral = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltRandomEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 3 )
)
process.hltPreHIRandom = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sETT2000 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETT2000" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUCC010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltTowerMakerForHf = cms.EDProducer( "CaloTowersCreator",
    EBSumThreshold = cms.double( 0.2 ),
    MomHBDepth = cms.double( 0.2 ),
    UseEtEBTreshold = cms.bool( False ),
    hfInput = cms.InputTag( "hltHfreco" ),
    AllowMissingInputs = cms.bool( True ),
    MomEEDepth = cms.double( 0.0 ),
    EESumThreshold = cms.double( 0.45 ),
    HBGrid = cms.vdouble(  ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    HBThreshold = cms.double( 0.7 ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  ),
    UseEcalRecoveredHits = cms.bool( True ),
    MomConstrMethod = cms.int32( 1 ),
    MomHEDepth = cms.double( 0.4 ),
    HcalThreshold = cms.double( -1000.0 ),
    HF2Weights = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    UseSymEBTreshold = cms.bool( False ),
    EEWeights = cms.vdouble(  ),
    EEWeight = cms.double( 1.0E-99 ),
    UseHO = cms.bool( False ),
    HBWeights = cms.vdouble(  ),
    HF1Weight = cms.double( 1.0 ),
    HF2Grid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    EBWeight = cms.double( 1.0E-99 ),
    HF1Grid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    HOWeight = cms.double( 1.0E-99 ),
    HESWeight = cms.double( 1.0 ),
    HESThreshold = cms.double( 0.8 ),
    hbheInput = cms.InputTag( "" ),
    HF2Weight = cms.double( 1.0 ),
    HF2Threshold = cms.double( 0.85 ),
    HcalAcceptSeverityLevel = cms.uint32( 11 ),
    EEThreshold = cms.double( 0.3 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HF1Weights = cms.vdouble(  ),
    hoInput = cms.InputTag( "" ),
    HF1Threshold = cms.double( 0.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HESGrid = cms.vdouble(  ),
    EcutTower = cms.double( -1000.0 ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    HESWeights = cms.vdouble(  ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kProblematic',
      'kRecovered',
      'kTime',
      'kWeird',
      'kBad' ),
    HEDWeight = cms.double( 1.0 ),
    UseSymEETreshold = cms.bool( False ),
    HEDThreshold = cms.double( 0.8 ),
    EBThreshold = cms.double( 0.07 ),
    UseRejectedHitsOnly = cms.bool( False ),
    UseHcalRecoveredHits = cms.bool( True ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HOThreshold0 = cms.double( 3.5 ),
    ecalInputs = cms.VInputTag(  ),
    UseRejectedRecoveredHcalHits = cms.bool( True ),
    MomEBDepth = cms.double( 0.3 ),
    HBWeight = cms.double( 1.0 ),
    HOGrid = cms.vdouble(  ),
    EBGrid = cms.vdouble(  )
)
process.hltMetForHf = cms.EDProducer( "METProducer",
    resolutionsEra = cms.string( "Spring10" ),
    HB_EtResPar = cms.vdouble( 0.0, 1.22, 0.05 ),
    EE_PhiResPar = cms.vdouble( 0.02511 ),
    jdpt9 = cms.vdouble( 0.843, 0.885, 1.245, 1.665, 1.944, 1.981, 1.972, 2.875, 3.923, 7.51 ),
    jdpt8 = cms.vdouble( 0.889, 0.939, 1.166, 1.365, 1.553, 1.805, 2.06, 2.22, 2.268, 2.247 ),
    jdpt7 = cms.vdouble( 1.094, 1.139, 1.436, 1.672, 1.831, 2.05, 2.267, 2.549, 2.785, 2.86 ),
    jdpt6 = cms.vdouble( 1.213, 1.298, 1.716, 2.015, 2.191, 2.612, 2.863, 2.879, 2.925, 2.902 ),
    jdpt5 = cms.vdouble( 1.049, 1.149, 1.607, 1.869, 2.012, 2.219, 2.289, 2.412, 2.695, 2.865 ),
    jdpt4 = cms.vdouble( 0.85, 0.961, 1.337, 1.593, 1.854, 2.005, 2.209, 2.533, 2.812, 3.047 ),
    jdpt3 = cms.vdouble( 0.929, 1.04, 1.46, 1.74, 2.042, 2.289, 2.639, 2.837, 2.946, 2.971 ),
    jdpt2 = cms.vdouble( 0.841, 0.937, 1.316, 1.605, 1.919, 2.295, 2.562, 2.722, 2.943, 3.293 ),
    jdpt1 = cms.vdouble( 0.718, 0.813, 1.133, 1.384, 1.588, 1.841, 2.115, 2.379, 2.508, 2.772 ),
    jdpt0 = cms.vdouble( 0.749, 0.829, 1.099, 1.355, 1.584, 1.807, 2.035, 2.217, 2.378, 2.591 ),
    HE_EtResPar = cms.vdouble( 0.0, 1.3, 0.05 ),
    alias = cms.string( "RawCaloMET" ),
    HF_PhiResPar = cms.vdouble( 0.05022 ),
    InputType = cms.string( "CandidateCollection" ),
    HE_PhiResPar = cms.vdouble( 0.02511 ),
    HB_PhiResPar = cms.vdouble( 0.02511 ),
    EE_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    noHF = cms.bool( False ),
    PF_PhiResType2 = cms.vdouble( 0.0020 ),
    PF_PhiResType3 = cms.vdouble( 0.0020 ),
    HF_EtResPar = cms.vdouble( 0.0, 1.82, 0.09 ),
    resolutionsAlgo = cms.string( "AK5PF" ),
    PF_PhiResType6 = cms.vdouble( 0.02511 ),
    PF_PhiResType7 = cms.vdouble( 0.02511 ),
    PF_PhiResType4 = cms.vdouble( 0.0028, 0.0, 0.0022 ),
    PF_PhiResType5 = cms.vdouble( 0.1, 0.1, 0.13 ),
    ptresolthreshold = cms.double( 10.0 ),
    METType = cms.string( "CaloMET" ),
    EB_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    PF_PhiResType1 = cms.vdouble( 0.0020 ),
    globalThreshold = cms.double( 0.5 ),
    EB_PhiResPar = cms.vdouble( 0.00502 ),
    src = cms.InputTag( "hltTowerMakerForHf" ),
    jdphi9 = cms.vdouble( 0.062, 0.059, 0.053, 0.047, 0.042, 0.045, 0.036, 0.032, 0.034, 0.044 ),
    jdphi8 = cms.vdouble( 0.059, 0.057, 0.051, 0.044, 0.038, 0.035, 0.037, 0.032, 0.028, 0.028 ),
    jdphi4 = cms.vdouble( 0.042, 0.042, 0.043, 0.042, 0.038, 0.036, 0.036, 0.033, 0.031, 0.031 ),
    jdphi3 = cms.vdouble( 0.042, 0.043, 0.044, 0.043, 0.041, 0.039, 0.039, 0.036, 0.034, 0.031 ),
    jdphi2 = cms.vdouble( 0.04, 0.04, 0.04, 0.04, 0.04, 0.038, 0.036, 0.035, 0.034, 0.033 ),
    jdphi1 = cms.vdouble( 0.034, 0.035, 0.035, 0.035, 0.035, 0.034, 0.031, 0.03, 0.029, 0.027 ),
    jdphi0 = cms.vdouble( 0.034, 0.034, 0.034, 0.034, 0.032, 0.031, 0.028, 0.027, 0.027, 0.027 ),
    jdphi7 = cms.vdouble( 0.077, 0.072, 0.059, 0.05, 0.045, 0.042, 0.039, 0.039, 0.037, 0.031 ),
    jdphi6 = cms.vdouble( 0.084, 0.08, 0.072, 0.065, 0.066, 0.06, 0.051, 0.049, 0.045, 0.045 ),
    jdphi5 = cms.vdouble( 0.069, 0.069, 0.064, 0.058, 0.053, 0.049, 0.049, 0.043, 0.039, 0.04 ),
    HO_EtResPar = cms.vdouble( 0.0, 1.3, 0.0050 ),
    HO_PhiResPar = cms.vdouble( 0.02511 ),
    PF_EtResType5 = cms.vdouble( 0.41, 0.52, 0.25 ),
    PF_EtResType4 = cms.vdouble( 0.042, 0.1, 0.0 ),
    PF_EtResType7 = cms.vdouble( 0.0, 1.22, 0.05 ),
    PF_EtResType6 = cms.vdouble( 0.0, 1.22, 0.05 ),
    PF_EtResType1 = cms.vdouble( 0.05, 0.0, 0.0 ),
    calculateSignificance = cms.bool( False ),
    PF_EtResType3 = cms.vdouble( 0.05, 0.0, 0.0 ),
    PF_EtResType2 = cms.vdouble( 0.05, 0.0, 0.0 ),
    usePt = cms.untracked.bool( True ),
    onlyFiducialParticles = cms.bool( False ),
    vertexRho = cms.double( 2.0 ),
    eVetoDeltaPhi = cms.double( 100.0 ),
    PFClustersHCAL = cms.InputTag( "particleFlowClusterHCAL" ),
    PFClustersHFHAD = cms.InputTag( "particleFlowClusterHFHAD" ),
    dupMinPt = cms.double( 0.0 ),
    chi2_tight_max = cms.double( 5.0 ),
    vertexZ = cms.double( 15.0 ),
    nLayersTight = cms.int32( 0 ),
    vertexNdof = cms.int32( 4 ),
    ptErr_max = cms.double( 0.2 ),
    corner = cms.double( 1.479 ),
    PFClustersECAL = cms.InputTag( "particleFlowClusterECAL" ),
    eta_max = cms.double( 2.65 ),
    muonInputTag = cms.InputTag( "muons" ),
    eVetoDeltaCotTheta = cms.double( 100.0 ),
    maxd0cut = cms.double( 0.3 ),
    PFClustersHFEM = cms.InputTag( "particleFlowClusterHFEM" ),
    d0cutb = cms.double( 0.5 ),
    checkTrackPropagation = cms.bool( False ),
    usePFClusters = cms.bool( False ),
    vertexMaxDZ = cms.double( 0.2 ),
    deltaRShower = cms.double( 0.01 ),
    chi2_max = cms.double( 5.0 ),
    maxpt_eta25 = cms.double( 0.0 ),
    track_algos = cms.vint32(  ),
    ptErr_tight_max = cms.double( 0.2 ),
    maxTrackAlgo = cms.int32( 8 ),
    nLayers = cms.int32( 0 ),
    correctShowerTracks = cms.bool( False ),
    vetoDuplicates = cms.bool( False ),
    pt_max = cms.double( 100.0 ),
    radius = cms.double( 130.0 ),
    nhits_tight_min = cms.double( 9.0 ),
    beamSpotInputTag = cms.InputTag( "unused" ),
    dupDCotTh = cms.double( 6.0E-4 ),
    usedeltaRRejection = cms.bool( False ),
    trackInputTag = cms.InputTag( "generalTracks" ),
    dupDPhi = cms.double( 0.03 ),
    electronInputTag = cms.InputTag( "gsfElectrons" ),
    tcmetDepValueMap = cms.InputTag( 'muonTCMETValueMapProducer','muCorrData' ),
    d0cuta = cms.double( 0.015 ),
    hOverECut = cms.double( 0.1 ),
    electronVetoCone = cms.bool( True ),
    muonDepValueMap = cms.InputTag( 'muonMETValueMapProducer','muCorrData' ),
    metInputTag = cms.InputTag( "met" ),
    usePvtxd0 = cms.bool( False ),
    vertexInputTag = cms.InputTag( "offlinePrimaryVertices" ),
    zdist = cms.double( 314.0 ),
    nhits_min = cms.double( 6.0 ),
    eVetoDeltaR = cms.double( 0.015 ),
    maxpt_eta20 = cms.double( 100.0 ),
    pt_min = cms.double( 1.0 ),
    rf_type = cms.int32( 0 ),
    nMinOuterHits = cms.int32( 2 ),
    track_quality = cms.vint32( 2 ),
    isCosmics = cms.bool( False ),
    eVetoMinElectronPt = cms.double( 10.0 )
)
process.hltGlobalSumETHfFilter3200 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 3200.0 ),
    Max = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hltPixelActivityFilter51500 = cms.EDFilter( "HLTPixelActivityFilter",
    maxClusters = cms.uint32( 1000000 ),
    saveTags = cms.bool( False ),
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 51500 )
)
process.hltPreHIUCC015 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltGlobalSumETHfFilter3100 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 3100.0 ),
    Max = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hltPixelActivityFilter51400 = cms.EDFilter( "HLTPixelActivityFilter",
    maxClusters = cms.uint32( 1000000 ),
    saveTags = cms.bool( False ),
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 51400 )
)
process.hltPreHICentral10 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltGlobalSumETHfFilter0 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 0.0 ),
    Max = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hltPixelActivityFilter0 = cms.EDFilter( "HLTPixelActivityFilter",
    maxClusters = cms.uint32( 1000000 ),
    saveTags = cms.bool( False ),
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 34000 )
)
process.hltL1sHIUPCNeuMu = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_MuOpen_ZdcCalo_NotBscMinBiasThresh2_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCNeuMuPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIUPCNeuEG2 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_EG2_ZdcCalo_NotBscMinBiasThresh2_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCNeuEG2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIUPCNeuEG5 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_EG5_ZdcCalo_NotBscMinBiasThresh2_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCNeuEG5PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIUPCNeuHcalHfMu = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_MuOpen_ZdcCalo_NotHcalHfCoincidencePm_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCNeuHcalHfMuPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIUPCNeuHcalHfEG2 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_EG2_ZdcCalo_NotHcalHfCoincidencePm_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCNeuHcalHfEG2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHIUPCNeuHcalHfEG5 = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_EG5_ZdcCalo_NotHcalHfCoincidencePm_BptxAND" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCNeuHcalHfEG5PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltFEDSelector = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1023 )
)
process.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
)
process.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
process.hltL1GtTrigReport = cms.EDAnalyzer( "L1GtTrigReport",
    PrintVerbosity = cms.untracked.int32( 10 ),
    UseL1GlobalTriggerRecord = cms.bool( False ),
    PrintOutput = cms.untracked.int32( 3 ),
    L1GtRecordInputTag = cms.InputTag( "hltGtDigis" )
)
process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)
process.hltPreAForHIOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreALCAP0Output = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreALCAPHISYMOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreALCALUMIPIXELSOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreCalibrationOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreDQMOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreEcalCalibrationOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreExpressForHIOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreExpressForHIOutputSmart = cms.EDFilter( "TriggerResultsFilter",
    l1tIgnoreMask = cms.bool( False ),
    l1tResults = cms.InputTag( "hltGtDigis" ),
    l1techIgnorePrescales = cms.bool( False ),
    hltResults = cms.InputTag( "TriggerResults" ),
    triggerConditions = cms.vstring( '( HLT_HIBptxXOR_v4 OR HLT_HICentral10_v7 OR HLT_HIL1Algo_BptxXOR_BSC_OR_v4 OR HLT_HIMinBiasBSC_OR_v4 OR HLT_HIMinBiasBSC_v4 OR HLT_HIMinBiasHF_v4 OR HLT_HIMinBiasHfOrBSC_v4 OR HLT_HIMinBiasHf_OR_v4 OR HLT_HIMinBiasPixel_SingleTrack_v5 OR HLT_HIMinBiasZDCPixel_SingleTrack_v5 OR HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4 OR HLT_HIMinBiasZDC_Calo_v4 OR HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5 OR HLT_HIPhysics_v4 OR HLT_HIRandom_v3 OR HLT_HIUCC010_v7 OR HLT_HIUCC015_v7 OR HLT_HIUPCNeuEG2Pixel_SingleTrack_v5 OR HLT_HIUPCNeuEG5Pixel_SingleTrack_v5 OR HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5 OR HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5 OR HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5 OR HLT_HIUPCNeuMuPixel_SingleTrack_v5 OR HLT_HIZeroBiasPixel_SingleTrack_v5 OR HLT_HIZeroBiasXOR_v4 OR HLT_HIZeroBias_v4 ) / 20',
      '( HLT_HIDiJet55_v7 OR HLT_HIDoublePhoton10_v5 OR HLT_HIDoublePhoton15_v5 OR HLT_HIDoublePhoton20_v5 OR HLT_HIFullTrack12_L1Central_v6 OR HLT_HIFullTrack12_L1Peripheral_v6 OR HLT_HIFullTrack14_L1Central_v6 OR HLT_HIFullTrack14_L1Peripheral_v6 OR HLT_HIFullTrack20_L1Central_v6 OR HLT_HIFullTrack20_L1Peripheral_v6 OR HLT_HIFullTrack25_L1Central_v6 OR HLT_HIFullTrack25_L1Peripheral_v6 OR HLT_HIJet55_v7 OR HLT_HIJet65_Jet55_v7 OR HLT_HIJet65_v7 OR HLT_HIJet80_v7 OR HLT_HIJet95_v7 OR HLT_HIJetE30_NoBPTX_v6 OR HLT_HIJetE50_NoBPTX3BX_NoHalo_v6 OR HLT_HIMET120_v6 OR HLT_HIMET200_v6 OR HLT_HIMET220_v6 OR HLT_HIPhoton10_Photon15_v5 OR HLT_HIPhoton15_Photon20_v5 OR HLT_HISinglePhoton15_v5 OR HLT_HISinglePhoton20_v6 OR HLT_HISinglePhoton30_v6 OR HLT_HISinglePhoton40_v6 ) / 20',
      '( HLT_HIL1DoubleMu0_HighQ_v5 OR HLT_HIL1DoubleMuOpen_v5 OR HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5 OR HLT_HIL2DoubleMu0_NHitQ_v5 OR HLT_HIL2DoubleMu0_v5 OR HLT_HIL2DoubleMu3_v5 OR HLT_HIL2Mu15_v5 OR HLT_HIL2Mu3_NHitQ_v5 OR HLT_HIL2Mu3_v5 OR HLT_HIL2Mu7_v5 OR HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7 OR HLT_HIL3DoubleMuOpen_Mgt2_OS_v7 OR HLT_HIL3DoubleMuOpen_Mgt2_SS_v7 OR HLT_HIL3DoubleMuOpen_Mgt2_v7 OR HLT_HIL3DoubleMuOpen_v7 OR HLT_HIL3Mu3_v7 ) / 20',
      '( HLT_HICentralityVeto_v5 ) / 2' ),
    throw = cms.bool( True ),
    daqPartitions = cms.uint32( 1 )
)
process.hltPreHLTDQMOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHLTDQMOutputSmart = cms.EDFilter( "TriggerResultsFilter",
    l1tIgnoreMask = cms.bool( False ),
    l1tResults = cms.InputTag( "hltGtDigis" ),
    l1techIgnorePrescales = cms.bool( False ),
    hltResults = cms.InputTag( "TriggerResults" ),
    triggerConditions = cms.vstring( 'HLT_HIMET120_v6',
      'HLT_HIMET200_v6',
      'HLT_HIMET220_v6',
      'HLT_HIPhysics_v4',
      'HLT_HIZeroBias_v4',
      'HLT_HIZeroBiasXOR_v4',
      'HLT_HIZeroBiasPixel_SingleTrack_v5',
      'HLT_HIMinBiasBSC_v4',
      'HLT_HIMinBiasBSC_OR_v4',
      'HLT_HIMinBiasHF_v4',
      'HLT_HIMinBiasHf_OR_v4',
      'HLT_HIMinBiasHfOrBSC_v4',
      'HLT_HIMinBiasPixel_SingleTrack_v5',
      'HLT_HIMinBiasZDC_Calo_v4',
      'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4',
      'HLT_HIMinBiasZDCPixel_SingleTrack_v5',
      'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5',
      'HLT_HIBptxXOR_v4',
      'HLT_HIL1Algo_BptxXOR_BSC_OR_v4',
      'HLT_HIL1DoubleMuOpen_v5',
      'HLT_HIL1DoubleMu0_HighQ_v5',
      'HLT_HIL2Mu3_v5',
      'HLT_HIL2Mu3_NHitQ_v5',
      'HLT_HIL2Mu7_v5',
      'HLT_HIL2Mu15_v5',
      'HLT_HIL2DoubleMu0_v5',
      'HLT_HIL2DoubleMu0_NHitQ_v5',
      'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5',
      'HLT_HIL2DoubleMu3_v5',
      'HLT_HIL3Mu3_v7',
      'HLT_HIL3DoubleMuOpen_v7',
      'HLT_HIL3DoubleMuOpen_Mgt2_v7',
      'HLT_HIL3DoubleMuOpen_Mgt2_SS_v7',
      'HLT_HIL3DoubleMuOpen_Mgt2_OS_v7',
      'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7',
      'HLT_HISinglePhoton15_v5',
      'HLT_HISinglePhoton20_v6',
      'HLT_HISinglePhoton30_v6',
      'HLT_HISinglePhoton40_v6',
      'HLT_HIPhoton10_Photon15_v5',
      'HLT_HIPhoton15_Photon20_v5',
      'HLT_HIDoublePhoton10_v5',
      'HLT_HIDoublePhoton15_v5',
      'HLT_HIDoublePhoton20_v5',
      'HLT_HIJet55_v7',
      'HLT_HIJet65_v7',
      'HLT_HIJet80_v7',
      'HLT_HIJet95_v7',
      'HLT_HIDiJet55_v7',
      'HLT_HIJet65_Jet55_v7',
      'HLT_HIJetE30_NoBPTX_v6',
      'HLT_HIJetE50_NoBPTX3BX_NoHalo_v6',
      'HLT_HIActivityHF_Coincidence3_v4',
      'HLT_HIActivityHF_Single3_v4',
      'HLT_HIClusterVertexCompatibility_v5',
      'HLT_HICentralityVeto_v5',
      'HLT_HIFullTrack12_L1Central_v6',
      'HLT_HIFullTrack12_L1Peripheral_v6',
      'HLT_HIFullTrack14_L1Central_v6',
      'HLT_HIFullTrack14_L1Peripheral_v6',
      'HLT_HIFullTrack20_L1Central_v6',
      'HLT_HIFullTrack20_L1Peripheral_v6',
      'HLT_HIFullTrack25_L1Central_v6',
      'HLT_HIFullTrack25_L1Peripheral_v6',
      'HLT_HIRandom_v3',
      'HLT_HIUCC010_v7',
      'HLT_HIUCC015_v7',
      'HLT_HICentral10_v7',
      'HLT_HIUPCNeuMuPixel_SingleTrack_v5',
      'HLT_HIUPCNeuEG2Pixel_SingleTrack_v5',
      'HLT_HIUPCNeuEG5Pixel_SingleTrack_v5',
      'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5',
      'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5',
      'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5' ),
    throw = cms.bool( True ),
    daqPartitions = cms.uint32( 1 )
)
process.hltPreHLTMONOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHLTMONOutputSmart = cms.EDFilter( "TriggerResultsFilter",
    l1tIgnoreMask = cms.bool( False ),
    l1tResults = cms.InputTag( "hltGtDigis" ),
    l1techIgnorePrescales = cms.bool( False ),
    hltResults = cms.InputTag( "TriggerResults" ),
    triggerConditions = cms.vstring( 'HLT_HIMET120_v6',
      'HLT_HIMET200_v6',
      'HLT_HIMET220_v6',
      'HLT_HIPhysics_v4',
      'HLT_HIZeroBias_v4',
      'HLT_HIZeroBiasXOR_v4',
      'HLT_HIZeroBiasPixel_SingleTrack_v5',
      'HLT_HIMinBiasBSC_v4',
      'HLT_HIMinBiasBSC_OR_v4',
      'HLT_HIMinBiasHF_v4',
      'HLT_HIMinBiasHf_OR_v4',
      'HLT_HIMinBiasHfOrBSC_v4',
      'HLT_HIMinBiasPixel_SingleTrack_v5',
      'HLT_HIMinBiasZDC_Calo_v4',
      'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4',
      'HLT_HIMinBiasZDCPixel_SingleTrack_v5',
      'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5',
      'HLT_HIBptxXOR_v4',
      'HLT_HIL1Algo_BptxXOR_BSC_OR_v4',
      'HLT_HIL1DoubleMuOpen_v5',
      'HLT_HIL1DoubleMu0_HighQ_v5',
      'HLT_HIL2Mu3_v5',
      'HLT_HIL2Mu3_NHitQ_v5',
      'HLT_HIL2Mu7_v5',
      'HLT_HIL2Mu15_v5',
      'HLT_HIL2DoubleMu0_v5',
      'HLT_HIL2DoubleMu0_NHitQ_v5',
      'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5',
      'HLT_HIL2DoubleMu3_v5',
      'HLT_HIL3Mu3_v7',
      'HLT_HIL3DoubleMuOpen_v7',
      'HLT_HIL3DoubleMuOpen_Mgt2_v7',
      'HLT_HIL3DoubleMuOpen_Mgt2_SS_v7',
      'HLT_HIL3DoubleMuOpen_Mgt2_OS_v7',
      'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7',
      'HLT_HISinglePhoton15_v5',
      'HLT_HISinglePhoton20_v6',
      'HLT_HISinglePhoton30_v6',
      'HLT_HISinglePhoton40_v6',
      'HLT_HIPhoton10_Photon15_v5',
      'HLT_HIPhoton15_Photon20_v5',
      'HLT_HIDoublePhoton10_v5',
      'HLT_HIDoublePhoton15_v5',
      'HLT_HIDoublePhoton20_v5',
      'HLT_HIJet55_v7',
      'HLT_HIJet65_v7',
      'HLT_HIJet80_v7',
      'HLT_HIJet95_v7',
      'HLT_HIDiJet55_v7',
      'HLT_HIJet65_Jet55_v7',
      'HLT_HIJetE30_NoBPTX_v6',
      'HLT_HIJetE50_NoBPTX3BX_NoHalo_v6',
      'HLT_HIActivityHF_Coincidence3_v4',
      'HLT_HIActivityHF_Single3_v4',
      'HLT_HIClusterVertexCompatibility_v5',
      'HLT_HICentralityVeto_v5',
      'HLT_HIFullTrack12_L1Central_v6',
      'HLT_HIFullTrack12_L1Peripheral_v6',
      'HLT_HIFullTrack14_L1Central_v6',
      'HLT_HIFullTrack14_L1Peripheral_v6',
      'HLT_HIFullTrack20_L1Central_v6',
      'HLT_HIFullTrack20_L1Peripheral_v6',
      'HLT_HIFullTrack25_L1Central_v6',
      'HLT_HIFullTrack25_L1Peripheral_v6',
      'HLT_HIRandom_v3',
      'HLT_HIUCC010_v7',
      'HLT_HIUCC015_v7',
      'HLT_HICentral10_v7',
      'HLT_HIUPCNeuMuPixel_SingleTrack_v5',
      'HLT_HIUPCNeuEG2Pixel_SingleTrack_v5',
      'HLT_HIUPCNeuEG5Pixel_SingleTrack_v5',
      'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5',
      'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5',
      'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5' ),
    throw = cms.bool( True ),
    daqPartitions = cms.uint32( 1 )
)
process.hltPreNanoDSTOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreRPCMONOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)

process.hltOutputAForHI = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputAForHI.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIBptxXOR_v4',
  'HLT_HICentral10_v7',
  'HLT_HICentralityVeto_v5',
  'HLT_HIDiJet55_v7',
  'HLT_HIDoublePhoton10_v5',
  'HLT_HIDoublePhoton15_v5',
  'HLT_HIDoublePhoton20_v5',
  'HLT_HIFullTrack12_L1Central_v6',
  'HLT_HIFullTrack12_L1Peripheral_v6',
  'HLT_HIFullTrack14_L1Central_v6',
  'HLT_HIFullTrack14_L1Peripheral_v6',
  'HLT_HIFullTrack20_L1Central_v6',
  'HLT_HIFullTrack20_L1Peripheral_v6',
  'HLT_HIFullTrack25_L1Central_v6',
  'HLT_HIFullTrack25_L1Peripheral_v6',
  'HLT_HIJet55_v7',
  'HLT_HIJet65_Jet55_v7',
  'HLT_HIJet65_v7',
  'HLT_HIJet80_v7',
  'HLT_HIJet95_v7',
  'HLT_HIJetE30_NoBPTX_v6',
  'HLT_HIJetE50_NoBPTX3BX_NoHalo_v6',
  'HLT_HIL1Algo_BptxXOR_BSC_OR_v4',
  'HLT_HIL1DoubleMu0_HighQ_v5',
  'HLT_HIL1DoubleMuOpen_v5',
  'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5',
  'HLT_HIL2DoubleMu0_NHitQ_v5',
  'HLT_HIL2DoubleMu0_v5',
  'HLT_HIL2DoubleMu3_v5',
  'HLT_HIL2Mu15_v5',
  'HLT_HIL2Mu3_NHitQ_v5',
  'HLT_HIL2Mu3_v5',
  'HLT_HIL2Mu7_v5',
  'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_OS_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_SS_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_v7',
  'HLT_HIL3DoubleMuOpen_v7',
  'HLT_HIL3Mu3_v7',
  'HLT_HIMET120_v6',
  'HLT_HIMET200_v6',
  'HLT_HIMET220_v6',
  'HLT_HIMinBiasBSC_OR_v4',
  'HLT_HIMinBiasBSC_v4',
  'HLT_HIMinBiasHF_v4',
  'HLT_HIMinBiasHfOrBSC_v4',
  'HLT_HIMinBiasHf_OR_v4',
  'HLT_HIMinBiasPixel_SingleTrack_v5',
  'HLT_HIMinBiasZDCPixel_SingleTrack_v5',
  'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4',
  'HLT_HIMinBiasZDC_Calo_v4',
  'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5',
  'HLT_HIPhoton10_Photon15_v5',
  'HLT_HIPhoton15_Photon20_v5',
  'HLT_HIPhysics_v4',
  'HLT_HIRandom_v3',
  'HLT_HISinglePhoton15_v5',
  'HLT_HISinglePhoton20_v6',
  'HLT_HISinglePhoton30_v6',
  'HLT_HISinglePhoton40_v6',
  'HLT_HIUCC010_v7',
  'HLT_HIUCC015_v7',
  'HLT_HIUPCNeuEG2Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuEG5Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5',
  'HLT_HIUPCNeuMuPixel_SingleTrack_v5',
  'HLT_HIZeroBiasPixel_SingleTrack_v5',
  'HLT_HIZeroBiasXOR_v4',
  'HLT_HIZeroBias_v4' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataRepacker_*_*',
      'keep FEDRawDataCollection_virginRawDataRepacker_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputCalibration.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIDTCalibration_v3',
  'HLT_HIEcalCalibration_v3',
  'HLT_HIHcalCalibration_v3' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltDTCalibrationRaw_*_*',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep *_hltHcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputEcalCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputEcalCalibration.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIEcalCalibration_v3' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputExpressForHI = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputExpressForHI.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIBptxXOR_v4',
  'HLT_HICentral10_v7',
  'HLT_HICentralityVeto_v5',
  'HLT_HIDiJet55_v7',
  'HLT_HIDoublePhoton10_v5',
  'HLT_HIDoublePhoton15_v5',
  'HLT_HIDoublePhoton20_v5',
  'HLT_HIFullTrack12_L1Central_v6',
  'HLT_HIFullTrack12_L1Peripheral_v6',
  'HLT_HIFullTrack14_L1Central_v6',
  'HLT_HIFullTrack14_L1Peripheral_v6',
  'HLT_HIFullTrack20_L1Central_v6',
  'HLT_HIFullTrack20_L1Peripheral_v6',
  'HLT_HIFullTrack25_L1Central_v6',
  'HLT_HIFullTrack25_L1Peripheral_v6',
  'HLT_HIJet55_v7',
  'HLT_HIJet65_Jet55_v7',
  'HLT_HIJet65_v7',
  'HLT_HIJet80_v7',
  'HLT_HIJet95_v7',
  'HLT_HIJetE30_NoBPTX_v6',
  'HLT_HIJetE50_NoBPTX3BX_NoHalo_v6',
  'HLT_HIL1Algo_BptxXOR_BSC_OR_v4',
  'HLT_HIL1DoubleMu0_HighQ_v5',
  'HLT_HIL1DoubleMuOpen_v5',
  'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5',
  'HLT_HIL2DoubleMu0_NHitQ_v5',
  'HLT_HIL2DoubleMu0_v5',
  'HLT_HIL2DoubleMu3_v5',
  'HLT_HIL2Mu15_v5',
  'HLT_HIL2Mu3_NHitQ_v5',
  'HLT_HIL2Mu3_v5',
  'HLT_HIL2Mu7_v5',
  'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_OS_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_SS_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_v7',
  'HLT_HIL3DoubleMuOpen_v7',
  'HLT_HIL3Mu3_v7',
  'HLT_HIMET120_v6',
  'HLT_HIMET200_v6',
  'HLT_HIMET220_v6',
  'HLT_HIMinBiasBSC_OR_v4',
  'HLT_HIMinBiasBSC_v4',
  'HLT_HIMinBiasHF_v4',
  'HLT_HIMinBiasHfOrBSC_v4',
  'HLT_HIMinBiasHf_OR_v4',
  'HLT_HIMinBiasPixel_SingleTrack_v5',
  'HLT_HIMinBiasZDCPixel_SingleTrack_v5',
  'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4',
  'HLT_HIMinBiasZDC_Calo_v4',
  'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5',
  'HLT_HIPhoton10_Photon15_v5',
  'HLT_HIPhoton15_Photon20_v5',
  'HLT_HIPhysics_v4',
  'HLT_HIRandom_v3',
  'HLT_HISinglePhoton15_v5',
  'HLT_HISinglePhoton20_v6',
  'HLT_HISinglePhoton30_v6',
  'HLT_HISinglePhoton40_v6',
  'HLT_HIUCC010_v7',
  'HLT_HIUCC015_v7',
  'HLT_HIUPCNeuEG2Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuEG5Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5',
  'HLT_HIUPCNeuMuPixel_SingleTrack_v5',
  'HLT_HIZeroBiasPixel_SingleTrack_v5',
  'HLT_HIZeroBiasXOR_v4',
  'HLT_HIZeroBias_v4' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataRepacker_*_*',
      'keep FEDRawDataCollection_virginRawDataRepacker_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputHLTDQM = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputHLTDQM.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIActivityHF_Coincidence3_v4',
  'HLT_HIActivityHF_Single3_v4',
  'HLT_HIBptxXOR_v4',
  'HLT_HICentral10_v7',
  'HLT_HICentralityVeto_v5',
  'HLT_HIClusterVertexCompatibility_v5',
  'HLT_HIDiJet55_v7',
  'HLT_HIDoublePhoton10_v5',
  'HLT_HIDoublePhoton15_v5',
  'HLT_HIDoublePhoton20_v5',
  'HLT_HIFullTrack12_L1Central_v6',
  'HLT_HIFullTrack12_L1Peripheral_v6',
  'HLT_HIFullTrack14_L1Central_v6',
  'HLT_HIFullTrack14_L1Peripheral_v6',
  'HLT_HIFullTrack20_L1Central_v6',
  'HLT_HIFullTrack20_L1Peripheral_v6',
  'HLT_HIFullTrack25_L1Central_v6',
  'HLT_HIFullTrack25_L1Peripheral_v6',
  'HLT_HIJet55_v7',
  'HLT_HIJet65_Jet55_v7',
  'HLT_HIJet65_v7',
  'HLT_HIJet80_v7',
  'HLT_HIJet95_v7',
  'HLT_HIJetE30_NoBPTX_v6',
  'HLT_HIJetE50_NoBPTX3BX_NoHalo_v6',
  'HLT_HIL1Algo_BptxXOR_BSC_OR_v4',
  'HLT_HIL1DoubleMu0_HighQ_v5',
  'HLT_HIL1DoubleMuOpen_v5',
  'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5',
  'HLT_HIL2DoubleMu0_NHitQ_v5',
  'HLT_HIL2DoubleMu0_v5',
  'HLT_HIL2DoubleMu3_v5',
  'HLT_HIL2Mu15_v5',
  'HLT_HIL2Mu3_NHitQ_v5',
  'HLT_HIL2Mu3_v5',
  'HLT_HIL2Mu7_v5',
  'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_OS_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_SS_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_v7',
  'HLT_HIL3DoubleMuOpen_v7',
  'HLT_HIL3Mu3_v7',
  'HLT_HIMET120_v6',
  'HLT_HIMET200_v6',
  'HLT_HIMET220_v6',
  'HLT_HIMinBiasBSC_OR_v4',
  'HLT_HIMinBiasBSC_v4',
  'HLT_HIMinBiasHF_v4',
  'HLT_HIMinBiasHfOrBSC_v4',
  'HLT_HIMinBiasHf_OR_v4',
  'HLT_HIMinBiasPixel_SingleTrack_v5',
  'HLT_HIMinBiasZDCPixel_SingleTrack_v5',
  'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4',
  'HLT_HIMinBiasZDC_Calo_v4',
  'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5',
  'HLT_HIPhoton10_Photon15_v5',
  'HLT_HIPhoton15_Photon20_v5',
  'HLT_HIPhysics_v4',
  'HLT_HIRandom_v3',
  'HLT_HISinglePhoton15_v5',
  'HLT_HISinglePhoton20_v6',
  'HLT_HISinglePhoton30_v6',
  'HLT_HISinglePhoton40_v6',
  'HLT_HIUCC010_v7',
  'HLT_HIUCC015_v7',
  'HLT_HIUPCNeuEG2Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuEG5Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5',
  'HLT_HIUPCNeuMuPixel_SingleTrack_v5',
  'HLT_HIZeroBiasPixel_SingleTrack_v5',
  'HLT_HIZeroBiasXOR_v4',
  'HLT_HIZeroBias_v4' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltTriggerSummaryAOD_*_*',
      'keep DcsStatuss_hltScalersRawToDigi_*_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep LumiScalerss_hltScalersRawToDigi_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputHLTMON = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputHLTMON.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIActivityHF_Coincidence3_v4',
  'HLT_HIActivityHF_Single3_v4',
  'HLT_HIBptxXOR_v4',
  'HLT_HICentral10_v7',
  'HLT_HICentralityVeto_v5',
  'HLT_HIClusterVertexCompatibility_v5',
  'HLT_HIDiJet55_v7',
  'HLT_HIDoublePhoton10_v5',
  'HLT_HIDoublePhoton15_v5',
  'HLT_HIDoublePhoton20_v5',
  'HLT_HIFullTrack12_L1Central_v6',
  'HLT_HIFullTrack12_L1Peripheral_v6',
  'HLT_HIFullTrack14_L1Central_v6',
  'HLT_HIFullTrack14_L1Peripheral_v6',
  'HLT_HIFullTrack20_L1Central_v6',
  'HLT_HIFullTrack20_L1Peripheral_v6',
  'HLT_HIFullTrack25_L1Central_v6',
  'HLT_HIFullTrack25_L1Peripheral_v6',
  'HLT_HIJet55_v7',
  'HLT_HIJet65_Jet55_v7',
  'HLT_HIJet65_v7',
  'HLT_HIJet80_v7',
  'HLT_HIJet95_v7',
  'HLT_HIJetE30_NoBPTX_v6',
  'HLT_HIJetE50_NoBPTX3BX_NoHalo_v6',
  'HLT_HIL1Algo_BptxXOR_BSC_OR_v4',
  'HLT_HIL1DoubleMu0_HighQ_v5',
  'HLT_HIL1DoubleMuOpen_v5',
  'HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5',
  'HLT_HIL2DoubleMu0_NHitQ_v5',
  'HLT_HIL2DoubleMu0_v5',
  'HLT_HIL2DoubleMu3_v5',
  'HLT_HIL2Mu15_v5',
  'HLT_HIL2Mu3_NHitQ_v5',
  'HLT_HIL2Mu3_v5',
  'HLT_HIL2Mu7_v5',
  'HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_OS_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_SS_v7',
  'HLT_HIL3DoubleMuOpen_Mgt2_v7',
  'HLT_HIL3DoubleMuOpen_v7',
  'HLT_HIL3Mu3_v7',
  'HLT_HIMET120_v6',
  'HLT_HIMET200_v6',
  'HLT_HIMET220_v6',
  'HLT_HIMinBiasBSC_OR_v4',
  'HLT_HIMinBiasBSC_v4',
  'HLT_HIMinBiasHF_v4',
  'HLT_HIMinBiasHfOrBSC_v4',
  'HLT_HIMinBiasHf_OR_v4',
  'HLT_HIMinBiasPixel_SingleTrack_v5',
  'HLT_HIMinBiasZDCPixel_SingleTrack_v5',
  'HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4',
  'HLT_HIMinBiasZDC_Calo_v4',
  'HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5',
  'HLT_HIPhoton10_Photon15_v5',
  'HLT_HIPhoton15_Photon20_v5',
  'HLT_HIPhysics_v4',
  'HLT_HIRandom_v3',
  'HLT_HISinglePhoton15_v5',
  'HLT_HISinglePhoton20_v6',
  'HLT_HISinglePhoton30_v6',
  'HLT_HISinglePhoton40_v6',
  'HLT_HIUCC010_v7',
  'HLT_HIUCC015_v7',
  'HLT_HIUPCNeuEG2Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuEG5Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5',
  'HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5',
  'HLT_HIUPCNeuMuPixel_SingleTrack_v5',
  'HLT_HIZeroBiasPixel_SingleTrack_v5',
  'HLT_HIZeroBiasXOR_v4',
  'HLT_HIZeroBias_v4' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltDoublePFTau25TrackPt5MediumIsolationProng4L1HLTMatched_*_*',
      'keep *_hltDoublePFTau25TrackPt5MediumIsolationProng4_*_*',
      'keep *_hltDoublePFTau25TrackPt5MediumIsolation_*_*',
      'keep *_hltDoublePFTau25TrackPt5_*_*',
      'keep *_hltDoublePFTau25_*_*',
      'keep *_hltEle20CaloIdVTTrkIdTDphiFilter_*_*',
      'keep *_hltIter1Merged_*_*',
      'keep *_hltIter2Merged_*_*',
      'keep *_hltIter3Merged_*_*',
      'keep *_hltIter4Merged_*_*',
      'keep *_hltL1extraParticlesCentral_*_*',
      'keep *_hltL1extraParticlesNonIsolated_*_*',
      'keep *_hltL1extraParticlesTau_*_*',
      'keep *_hltL1extraParticles_*_*',
      'keep *_hltL1sDoubleTauJet44erorDoubleJetC64_*_*',
      'keep *_hltL1sL1EG18er_*_*',
      'keep *_hltL1sL1ETM36ORETM40_*_*',
      'keep *_hltL1sMu16Eta2p1_*_*',
      'keep *_hltL3TkTracksFromL2_*_*',
      'keep *_hltL3crIsoL1sMu16Eta2p1L1f0L2f16QL3f18QL3crIsoRhoFiltered0p15_*_*',
      'keep *_hltOverlapFilterEle20LooseIsoPFTau20OldVersion_*_*',
      'keep *_hltOverlapFilterIsoMu18LooseIsoPFTau20_*_*',
      'keep *_hltOverlapFilterIsoMu18PFTau25TrackPt5Prong4_*_*',
      'keep *_hltPFTau20IsoMuVertex_*_*',
      'keep *_hltPFTau20TrackLooseIso_*_*',
      'keep *_hltPFTau20Track_*_*',
      'keep *_hltPFTau20_*_*',
      'keep *_hltPFTau25TrackPt5MediumIsolationProng4IsoMuVertex_*_*',
      'keep *_hltPFTau25TrackPt5MediumIsolationProng4_*_*',
      'keep *_hltPFTau25TrackPt5MediumIsolation_*_*',
      'keep *_hltPFTau25TrackPt5_*_*',
      'keep *_hltPFTau25_*_*',
      'keep *_hltPFTau35TrackPt20LooseIsoProng2_*_*',
      'keep *_hltPFTau35TrackPt20LooseIso_*_*',
      'keep *_hltPFTau35TrackPt20_*_*',
      'keep *_hltPFTau35Track_*_*',
      'keep *_hltPFTau35_*_*',
      'keep *_hltPFTauEleVertex20_*_*',
      'keep *_hltPixelTracks_*_*',
      'keep *_hltPixelVertices3DbbPhi_*_*',
      'keep *_hltPixelVertices_*_*',
      'keep *_hltTriggerSummaryAOD_*_*',
      'keep *_hltTriggerSummaryRAW_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep FEDRawDataCollection_rawDataRepacker_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep recoCaloJets_*_*_*',
      'keep recoCaloMETs_*_*_*',
      'keep recoCompositeCandidates_*_*_*',
      'keep recoElectrons_*_*_*',
      'keep recoIsolatedPixelTrackCandidates_*_*_*',
      'keep recoMETs_*_*_*',
      'keep recoPFJets_*_*_*',
      'keep recoPFTaus_*_*_*',
      'keep recoRecoChargedCandidates_*_*_*',
      'keep recoRecoEcalCandidates_*_*_*',
      'keep triggerTriggerEventWithRefs_*_*_*',
      'keep triggerTriggerEvent_*_*_*',
      'keep triggerTriggerFilterObjectWithRefs_*_*_*' )
)

process.HLTL1UnpackerSequence = cms.Sequence( process.hltGtDigis + process.hltGctDigis + process.hltL1GtObjectMap + process.hltL1extraParticles )
process.HLTBeamSpot = cms.Sequence( process.hltScalersRawToDigi + process.hltOnlineBeamSpot )
process.HLTBeginSequence = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.HLTBeamSpot )
process.HLTDoLocalHcalSequence = cms.Sequence( process.hltHcalDigis + process.hltHbhereco + process.hltHfreco + process.hltHoreco )
process.HLTDoCaloSequence = cms.Sequence( process.hltEcalRawToRecHitFacility + process.hltEcalRegionalRestFEDs + process.hltEcalRecHitAll + process.HLTDoLocalHcalSequence + process.hltTowerMakerForAll )
process.HLTRecoMETSequence = cms.Sequence( process.HLTDoCaloSequence + process.hltMet )
process.HLTDoHIStripZeroSuppression = cms.Sequence( process.hltSiStripRawToDigi + process.hltSiStripZeroSuppression + process.hltSiStripDigiToZSRaw + process.hltSiStripRawDigiToVirginRaw + process.virginRawDataRepacker + process.rawDataRepacker )
process.HLTEndSequence = cms.Sequence( process.hltBoolEnd )
process.HLTBeginSequenceCalibration = cms.Sequence( process.hltCalibrationEventsFilter + process.hltGtDigis )
process.HLTDoHILocalPixelSequence = cms.Sequence( process.hltSiPixelDigis + process.hltHISiPixelClusters + process.hltHISiPixelRecHits )
process.HLTPixelTrackingForHITrackTrigger = cms.Sequence( process.hltHIPixelClusterVertices + process.hltPixelTracksForHITrackTrigger + process.hltPixelCandsForHITrackTrigger )
process.HLTMuonLocalRecoSequence = cms.Sequence( process.hltMuonDTDigis + process.hltDt1DRecHits + process.hltDt4DSegments + process.hltMuonCSCDigis + process.hltCsc2DRecHits + process.hltCscSegments + process.hltMuonRPCDigis + process.hltRpcRecHits )
process.HLTL2muonrecoNocandSequence = cms.Sequence( process.HLTMuonLocalRecoSequence + process.hltL2OfflineMuonSeeds + process.hltL2MuonSeeds + process.hltL2Muons )
process.HLTL2muonrecoSequence = cms.Sequence( process.HLTL2muonrecoNocandSequence + process.hltL2MuonCandidates )
process.HLTDoHILocalStripSequence = cms.Sequence( process.hltSiStripExcludedFEDListProducer + process.hltHISiStripRawToClustersFacility + process.hltHISiStripClusters )
process.HLTHIL3muonTkCandidateSequence = cms.Sequence( process.HLTDoHILocalPixelSequence + process.HLTDoHILocalStripSequence + process.hltL3TrajSeedOIState + process.hltHIL3TrackCandidateFromL2OIState + process.hltHIL3TkTracksFromL2OIState + process.hltHIL3MuonsOIState + process.hltHIL3TrajSeedOIHit + process.hltHIL3TrackCandidateFromL2OIHit + process.hltHIL3TkTracksFromL2OIHit + process.hltHIL3MuonsOIHit + process.hltHIL3TkFromL2OICombination + process.hltHIL3TrajSeedIOHit + process.hltHIL3TrackCandidateFromL2IOHit + process.hltHIL3TkTracksFromL2IOHit + process.hltHIAllL3MuonsIOHit + process.hltHIL3TrajectorySeed + process.hltHIL3TrackCandidateFromL2 )
process.HLTHIL3muonrecoNocandSequence = cms.Sequence( process.HLTHIL3muonTkCandidateSequence + process.hltHIL3TkTracksFromL2 + process.hltHIL3MuonsLinksCombination + process.hltHIL3Muons )
process.HLTHIL3muonrecoSequence = cms.Sequence( process.HLTHIL3muonrecoNocandSequence + process.hltHIL3MuonCandidates )
process.HLTDoHIEcalClusWithCleaningSequence = cms.Sequence( process.hltIslandBasicClustersHI + process.hltHiIslandSuperClustersHI + process.hltHiCorrectedIslandBarrelSuperClustersHI + process.hltHiCorrectedIslandEndcapSuperClustersHI + process.hltCleanedHiCorrectedIslandBarrelSuperClustersHI + process.hltRecoHIEcalWithCleaningCandidate )
process.HLTHIRecoJetSequenceIC5Uncorrected = cms.Sequence( process.HLTDoCaloSequence + process.hltIterativeCone5PileupSubtractionCaloJets )
process.HLTHIRecoJetSequenceIC5Corrected = cms.Sequence( process.HLTHIRecoJetSequenceIC5Uncorrected + process.hltHICaloJetIDPassed + process.hltHICaloJetCorrected )
process.HLTStoppedHSCPLocalHcalReco = cms.Sequence( process.hltHcalDigis + process.hltHbhereco )
process.HLTStoppedHSCPJetSequence = cms.Sequence( process.hltStoppedHSCPTowerMakerForAll + process.hltStoppedHSCPIterativeCone5CaloJets )
process.HLTBeginSequenceAntiBPTX = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.hltBPTXAntiCoincidence + process.HLTBeamSpot )
process.HLTBeginSequenceBPTX = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.hltBPTXCoincidence + process.HLTBeamSpot )
process.HLTPixelSeedingForHITrackTrigger = cms.Sequence( process.hltHIPixelClusterVerticesForHITrackTrigger + process.hltHIPixel3ProtoTracks + process.hltHIPixelMedianVertex + process.hltHISelectedProtoTracks + process.hltHIPixelAdaptiveVertex + process.hltHIBestAdaptiveVertex + process.hltHISelectedVertex + process.hltHIPixel3PrimTracks + process.hltHIPixelTrackCandsForHITrackTrigger )
process.HLTDoHILocalStripSequenceNonRegional = cms.Sequence( process.hltSiStripRawToDigi + process.hltSiStripZeroSuppression + process.hltHISiStripClustersNonRegional )
process.HLTFullTrackingForHITrackTrigger = cms.Sequence( process.hltHIPixelTrackSeeds + process.hltHIPrimTrackCandidates + process.hltHIGlobalPrimTracks + process.hltHIGoodLooseTracks + process.hltHIFullTrackCandsForHITrackTrigger )
process.HLTDoLocalHfSequence = cms.Sequence( process.hltHcalDigis + process.hltHfreco + process.hltTowerMakerForHf )
process.HLTRecoMETHfSequence = cms.Sequence( process.HLTDoLocalHfSequence + process.hltMetForHf )
process.HLTDoHILocalPixelClustersSequence = cms.Sequence( process.hltSiPixelDigis + process.hltHISiPixelClusters )

process.HLTriggerFirstPath = cms.Path( process.hltGetConditions + process.hltGetRaw + process.hltBoolFalse )
process.HLT_HIMET120_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ETM30BptxAND + process.hltPreHIMET120 + process.HLTRecoMETSequence + process.hltHIMET120 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMET200_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ETM50BptxAND + process.hltPreHIMET200 + process.HLTRecoMETSequence + process.hltHIMET200 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMET220_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ETM50BptxAND + process.hltPreHIMET220 + process.HLTRecoMETSequence + process.hltHIMET220 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPhysics_v4 = cms.Path( process.HLTBeginSequence + process.hltPreHIPhysics + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIDTCalibration_v3 = cms.Path( process.HLTBeginSequenceCalibration + process.hltPreHIDTCalibration + process.hltDTCalibrationRaw + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIEcalCalibration_v3 = cms.Path( process.HLTBeginSequenceCalibration + process.hltPreHIEcalCalibration + process.hltEcalCalibrationRaw + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIHcalCalibration_v3 = cms.Path( process.HLTBeginSequenceCalibration + process.hltPreHIHcalCalibration + process.hltHcalCalibTypeFilter + process.hltHcalCalibrationRaw + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIZeroBias_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHIZeroBias + process.hltPreHIZeroBias + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIZeroBiasXOR_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sL1BptxXOR + process.hltPreHIZeroBiasXOR + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIZeroBiasPixel_SingleTrack_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sHIZeroBiasXOR + process.hltPreHIZeroBiasPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMinBiasBSC_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasBSC + process.hltPreHIMinBiasBSC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMinBiasBSC_OR_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasBSCOR + process.hltPreHIMinBiasBSCOR + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMinBiasHF_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasHF + process.hltPreHIMinBiasHF + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMinBiasHf_OR_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasHfOr + process.hltPreHIMinBiasHfOR + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMinBiasHfOrBSC_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasHfOrBSC + process.hltPreHIMinBiasHfOrBSC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMinBiasPixel_SingleTrack_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasHfOrBSC + process.hltPreHIMinBiasPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMinBiasZDC_Calo_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasZDC + process.hltPreHIMinBiasZDCCalo + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMinBiasZDC_Calo_PlusOrMinus_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasZDCCaloPlusOrMinus + process.hltPreHIMinBiasZDCCaloPlusOrMinus + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMinBiasZDCPixel_SingleTrack_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasZDCPixelSingleTrack + process.hltPreHIMinBiasZDCPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIMinBiasZDC_PlusOrMinusPixel_SingleTrack_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasZDCPlusOrMinusPixelSingleTrack + process.hltPreHIMinBiasZDCPlusOrMinusPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIBptxXOR_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sL1BptxXOR + process.hltPreHIBptxXOR + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1Algo_BptxXOR_BSC_OR_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sL1BptxXORBscMinBiasOR + process.hltPreHIL1AlgoBptxXORBSCOR + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1DoubleMuOpen_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenBptxAND + process.hltPreHIL1DoubleMuOpen + process.hltHIDoubleMuLevel1PathL1OpenFiltered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1DoubleMu0_HighQ_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenBptxAND + process.hltPreHIL1DoubleMu0HighQ + process.hltHIDoubleMuLevel1PathL1HighQFiltered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3BptxAND + process.hltPreHIL2Mu3 + process.hltHIL1SingleMu3withBptxANDL1Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3_NHitQ_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3BptxAND + process.hltPreHIL2Mu3NHitQ + process.hltHIL1SingleMu3withBptxANDL1Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3NHitL2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu7_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3BptxAND + process.hltPreHIL2Mu7 + process.hltHIL1SingleMu3withBptxANDL1Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu7L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu15_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3BptxAND + process.hltPreHIL2Mu15 + process.hltHIL1SingleMu3withBptxANDL1Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu15L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2DoubleMu0_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenBptxAND + process.hltPreHIL2DoubleMu0 + process.hltHIDoubleMuLevel1PathL1OpenFiltered + process.HLTL2muonrecoSequence + process.hltHIL2DoubleMu0L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2DoubleMu0_NHitQ_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenBptxAND + process.hltPreHIL2DoubleMu0NHitQ + process.hltHIDoubleMuLevel1PathL1OpenFiltered + process.HLTL2muonrecoSequence + process.hltHIL2DoubleMu0L2N1HitsFiltered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2DoubleMu0_L1HighQL2NHitQ_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenBptxAND + process.hltPreHIL2DoubleMu0L1HighQL2NHitQ + process.hltHIDoubleMuLevel1PathL1HighQFiltered + process.HLTL2muonrecoSequence + process.hltHIL2DoubleMu0L2N1HitsFiltered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2DoubleMu3_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenBptxAND + process.hltPreHIL2DoubleMu3 + process.hltHIDoubleMuLevel1PathL1OpenFiltered + process.HLTL2muonrecoSequence + process.hltHIL2DoubleMu3L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu3_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3BptxAND + process.hltPreHIL3Mu3 + process.hltHIL1SingleMu3withBptxANDL1Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3L2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu3L3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3DoubleMuOpen_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenBptxAND + process.hltPreHIL3DoubleMuOpen + process.hltHIDoubleMuLevel1PathL1OpenFiltered + process.HLTL2muonrecoSequence + process.hltHIDimuonL2PreFiltered0 + process.HLTHIL3muonrecoSequence + process.hltHIDimuonL3FilteredOpen + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3DoubleMuOpen_Mgt2_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenBptxAND + process.hltPreHIL3DoubleMuOpenMgt2 + process.hltHIDoubleMuLevel1PathL1OpenFiltered + process.HLTL2muonrecoSequence + process.hltHIDimuonL2PreFiltered0 + process.HLTHIL3muonrecoSequence + process.hltHIDimuonL3FilteredMgt2 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3DoubleMuOpen_Mgt2_SS_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenBptxAND + process.hltPreHIL3DoubleMuOpenMgt2SS + process.hltHIDoubleMuLevel1PathL1OpenFiltered + process.HLTL2muonrecoSequence + process.hltHIDimuonL2PreFiltered0 + process.HLTHIL3muonrecoSequence + process.hltHIDimuonL3FilteredMgt2SS + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3DoubleMuOpen_Mgt2_OS_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenBptxAND + process.hltPreHIL3DoubleMuOpenMgt2OS + process.hltHIDoubleMuLevel1PathL1OpenFiltered + process.HLTL2muonrecoSequence + process.hltHIDimuonL2PreFiltered0 + process.HLTHIL3muonrecoSequence + process.hltHIDimuonL3FilteredMg2OS + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3DoubleMuOpen_Mgt2_OS_NoCowboy_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenBptxAND + process.hltPreHIL3DoubleMuOpenMgt2OSNoCowboy + process.hltHIDoubleMuLevel1PathL1OpenFiltered + process.HLTL2muonrecoSequence + process.hltHIDimuonL2PreFiltered0 + process.HLTHIL3muonrecoSequence + process.hltHIDimuonL3FilteredMg2OSnoCowboy + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton15_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG5BptxAND + process.hltPreHISinglePhoton15 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusWithCleaningSequence + process.hltHIPhoton15 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton20_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG5BptxAND + process.hltPreHISinglePhoton20 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusWithCleaningSequence + process.hltHIPhoton20 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton30_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG5BptxAND + process.hltPreHISinglePhoton30 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusWithCleaningSequence + process.hltHIPhoton30 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton40_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG5BptxAND + process.hltPreHISinglePhoton40 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusWithCleaningSequence + process.hltHIPhoton40 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPhoton10_Photon15_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5BptxAND + process.hltPreHIPhoton10Photon15 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusWithCleaningSequence + process.hltHIDoublePhoton1015Filter1 + process.hltHIDoublePhoton1015Filter2 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPhoton15_Photon20_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5BptxAND + process.hltPreHIPhoton15Photon20 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusWithCleaningSequence + process.hltHIDoublePhoton1520Filter1 + process.hltHIDoublePhoton1520Filter2 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIDoublePhoton10_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5BptxAND + process.hltPreHIDoublePhoton10 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusWithCleaningSequence + process.hltHIDoublePhoton10 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIDoublePhoton15_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5BptxAND + process.hltPreHIDoublePhoton15 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusWithCleaningSequence + process.hltHIDoublePhoton15 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIDoublePhoton20_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG5BptxAND + process.hltPreHIDoublePhoton20 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusWithCleaningSequence + process.hltHIDoublePhoton20 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIJet55_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet36BptxAND + process.hltPreHIJet55 + process.HLTHIRecoJetSequenceIC5Corrected + process.hltHI1jet55 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIJet65_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet36BptxAND + process.hltPreHIJet65 + process.HLTHIRecoJetSequenceIC5Corrected + process.hltHI1jet65 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIJet80_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet52BptxAND + process.hltPreHIJet80 + process.HLTHIRecoJetSequenceIC5Corrected + process.hltHI1jet80 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIJet95_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet68BptxAND + process.hltPreHIJet95 + process.HLTHIRecoJetSequenceIC5Corrected + process.hltHI1jet95 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIDiJet55_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet36BptxAND + process.hltPreHIDiJet55 + process.HLTHIRecoJetSequenceIC5Corrected + process.hltHIDiJet55and55 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIJet65_Jet55_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet36BptxAND + process.hltPreHIJet65Jet55 + process.HLTHIRecoJetSequenceIC5Corrected + process.hltHIDiJet55and55 + process.hltHI1jet65 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIJetE30_NoBPTX_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet20CentralNotBptx + process.hltPreHIJetE30NoBPTX + process.HLTStoppedHSCPLocalHcalReco + process.HLTStoppedHSCPJetSequence + process.hltStoppedHSCP1CaloJetEnergy30 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIJetE50_NoBPTX3BX_NoHalo_v6 = cms.Path( process.HLTBeginSequenceAntiBPTX + process.hltL1sL1SingleJet32CentralNotBptx + process.hltL1BeamHaloAntiCoincidence3BX + process.hltPreHIJetE50NoBPTX3BXNoHalo + process.HLTStoppedHSCPLocalHcalReco + process.hltStoppedHSCPHpdFilter + process.HLTStoppedHSCPJetSequence + process.hltStoppedHSCP1CaloJetEnergy50 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIActivityHF_Coincidence3_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sL1GlobalDecision + process.hltPreHIActivityHFCoincidence3 + process.hltHcalDigis + process.hltHfreco + process.hltHcalSimpleRecHitFilterCoincidence + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIActivityHF_Single3_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sL1GlobalDecision + process.hltPreHIActivityHFSingle3 + process.hltHcalDigis + process.hltHfreco + process.hltHcalSimpleRecHitFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIClusterVertexCompatibility_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sL1GlobalDecision + process.hltPreHIClusterVertexCompatibility + process.HLTDoHILocalPixelSequence + process.hltHIPixelClusterShapeFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HICentralityVeto_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasHfOrBSC + process.hltPreHICentralityVeto + process.HLTDoHILocalPixelSequence + process.hltPixelActivityFilterCentralityVeto + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIFullTrack12_L1Central_v6 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sETT100BptxAND + process.hltPreHIFullTrack12L1Central + process.HLTDoCaloSequence + process.hltHICaloTowerFilter4 + process.HLTDoHILocalPixelSequence + process.HLTPixelSeedingForHITrackTrigger + process.hltHISinglePixelTrackFilter10 + process.HLTDoHILocalStripSequenceNonRegional + process.HLTFullTrackingForHITrackTrigger + process.hltHISingleFullTrackFilter12 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIFullTrack12_L1Peripheral_v6 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sJet16CentralNotETT140BptxAND + process.hltPreHIFullTrack12L1Peripheral + process.HLTDoCaloSequence + process.hltHICaloTowerFilter4 + process.HLTDoHILocalPixelSequence + process.HLTPixelSeedingForHITrackTrigger + process.hltHISinglePixelTrackFilter10 + process.HLTDoHILocalStripSequenceNonRegional + process.HLTFullTrackingForHITrackTrigger + process.hltHISingleFullTrackFilter12 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIFullTrack14_L1Central_v6 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sETT100BptxAND + process.hltPreHIFullTrack14L1Central + process.HLTDoCaloSequence + process.hltHICaloTowerFilter4 + process.HLTDoHILocalPixelSequence + process.HLTPixelSeedingForHITrackTrigger + process.hltHISinglePixelTrackFilter10 + process.HLTDoHILocalStripSequenceNonRegional + process.HLTFullTrackingForHITrackTrigger + process.hltHISingleFullTrackFilter14 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIFullTrack14_L1Peripheral_v6 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sJet16CentralNotETT140BptxAND + process.hltPreHIFullTrack14L1Peripheral + process.HLTDoCaloSequence + process.hltHICaloTowerFilter4 + process.HLTDoHILocalPixelSequence + process.HLTPixelSeedingForHITrackTrigger + process.hltHISinglePixelTrackFilter10 + process.HLTDoHILocalStripSequenceNonRegional + process.HLTFullTrackingForHITrackTrigger + process.hltHISingleFullTrackFilter14 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIFullTrack20_L1Central_v6 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sETT100BptxAND + process.hltPreHIFullTrack20L1Central + process.HLTDoCaloSequence + process.hltHICaloTowerFilter4 + process.HLTDoHILocalPixelSequence + process.HLTPixelSeedingForHITrackTrigger + process.hltHISinglePixelTrackFilter10 + process.HLTDoHILocalStripSequenceNonRegional + process.HLTFullTrackingForHITrackTrigger + process.hltHISingleFullTrackFilter20 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIFullTrack20_L1Peripheral_v6 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sJet16CentralNotETT140BptxAND + process.hltPreHIFullTrack20L1Peripheral + process.HLTDoCaloSequence + process.hltHICaloTowerFilter4 + process.HLTDoHILocalPixelSequence + process.HLTPixelSeedingForHITrackTrigger + process.hltHISinglePixelTrackFilter10 + process.HLTDoHILocalStripSequenceNonRegional + process.HLTFullTrackingForHITrackTrigger + process.hltHISingleFullTrackFilter20 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIFullTrack25_L1Central_v6 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sETT100BptxAND + process.hltPreHIFullTrack25L1Central + process.HLTDoCaloSequence + process.hltHICaloTowerFilter4 + process.HLTDoHILocalPixelSequence + process.HLTPixelSeedingForHITrackTrigger + process.hltHISinglePixelTrackFilter10 + process.HLTDoHILocalStripSequenceNonRegional + process.HLTFullTrackingForHITrackTrigger + process.hltHISingleFullTrackFilter25 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIFullTrack25_L1Peripheral_v6 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sJet16CentralNotETT140BptxAND + process.hltPreHIFullTrack25L1Peripheral + process.HLTDoCaloSequence + process.hltHICaloTowerFilter4 + process.HLTDoHILocalPixelSequence + process.HLTPixelSeedingForHITrackTrigger + process.hltHISinglePixelTrackFilter10 + process.HLTDoHILocalStripSequenceNonRegional + process.HLTFullTrackingForHITrackTrigger + process.hltHISingleFullTrackFilter25 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIRandom_v3 = cms.Path( process.hltRandomEventsFilter + process.HLTL1UnpackerSequence + process.hltPreHIRandom + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUCC010_v7 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sETT2000 + process.hltPreHIUCC010 + process.HLTRecoMETHfSequence + process.hltGlobalSumETHfFilter3200 + process.HLTDoHILocalPixelClustersSequence + process.hltPixelActivityFilter51500 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUCC015_v7 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sETT2000 + process.hltPreHIUCC015 + process.HLTRecoMETHfSequence + process.hltGlobalSumETHfFilter3100 + process.HLTDoHILocalPixelClustersSequence + process.hltPixelActivityFilter51400 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HICentral10_v7 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sETT2000 + process.hltPreHICentral10 + process.HLTRecoMETHfSequence + process.hltGlobalSumETHfFilter0 + process.HLTDoHILocalPixelClustersSequence + process.hltPixelActivityFilter0 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCNeuMuPixel_SingleTrack_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sHIUPCNeuMu + process.hltPreHIUPCNeuMuPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCNeuEG2Pixel_SingleTrack_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sHIUPCNeuEG2 + process.hltPreHIUPCNeuEG2PixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCNeuEG5Pixel_SingleTrack_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sHIUPCNeuEG5 + process.hltPreHIUPCNeuEG5PixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCNeuHcalHfMuPixel_SingleTrack_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sHIUPCNeuHcalHfMu + process.hltPreHIUPCNeuHcalHfMuPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCNeuHcalHfEG2Pixel_SingleTrack_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sHIUPCNeuHcalHfEG2 + process.hltPreHIUPCNeuHcalHfEG2PixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCNeuHcalHfEG5Pixel_SingleTrack_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sHIUPCNeuHcalHfEG5 + process.hltPreHIUPCNeuHcalHfEG5PixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLTriggerFinalPath = cms.Path( process.hltGtDigis + process.hltScalersRawToDigi + process.hltFEDSelector + process.hltTriggerSummaryAOD + process.hltTriggerSummaryRAW )
process.HLTAnalyzerEndpath = cms.EndPath( process.hltL1GtTrigReport + process.hltTrigReport )
process.AForHIOutput = cms.EndPath( process.hltPreAForHIOutput + process.hltOutputAForHI )
process.ALCAP0Output = cms.EndPath( process.hltPreALCAP0Output )
process.ALCAPHISYMOutput = cms.EndPath( process.hltPreALCAPHISYMOutput )
process.ALCALUMIPIXELSOutput = cms.EndPath( process.hltPreALCALUMIPIXELSOutput )
process.CalibrationOutput = cms.EndPath( process.hltPreCalibrationOutput + process.hltOutputCalibration )
process.DQMOutput = cms.Path( process.hltPreDQMOutput )
process.EcalCalibrationOutput = cms.EndPath( process.hltPreEcalCalibrationOutput + process.hltOutputEcalCalibration )
process.ExpressForHIOutput = cms.EndPath( process.hltPreExpressForHIOutput + process.hltPreExpressForHIOutputSmart + process.hltOutputExpressForHI )
process.HLTDQMOutput = cms.EndPath( process.hltPreHLTDQMOutput + process.hltPreHLTDQMOutputSmart + process.hltOutputHLTDQM )
process.HLTMONOutput = cms.EndPath( process.hltPreHLTMONOutput + process.hltPreHLTMONOutputSmart + process.hltOutputHLTMON )
process.NanoDSTOutput = cms.EndPath( process.hltPreNanoDSTOutput )
process.RPCMONOutput = cms.EndPath( process.hltPreRPCMONOutput )


process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        'file:RelVal_Raw_HIon_DATA.root',
    ),
    secondaryFileNames = cms.untracked.vstring(
    ),
    inputCommands = cms.untracked.vstring(
        'keep *'
    )
)

# Disable HF Noise filters in HIon menu
if 'hltHfreco' in process.__dict__:
    process.hltHfreco.setNoiseFlags = cms.bool( False )

# remove the HLT prescales
if 'PrescaleService' in process.__dict__:
    process.PrescaleService.lvl1DefaultLabel = cms.string( '0' )
    process.PrescaleService.lvl1Labels       = cms.vstring( '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' )
    process.PrescaleService.prescaleTable    = cms.VPSet( )

# CMSSW version specific customizations
import os
cmsswVersion = os.environ['CMSSW_VERSION']

# customization for CMSSW_5_2_X
if cmsswVersion.startswith('CMSSW_5_2_'):

    # force the use of the correct calo jet energy corrections
    if 'hltESPL1FastJetCorrectionESProducer' in process.__dict__:
        process.hltESPL1FastJetCorrectionESProducer.algorithm  = "AK5CaloHLT"

    if 'hltESPL2RelativeCorrectionESProducer' in process.__dict__:
        process.hltESPL2RelativeCorrectionESProducer.algorithm = "AK5CaloHLT"

    if 'hltESPL3AbsoluteCorrectionESProducer' in process.__dict__:
        process.hltESPL3AbsoluteCorrectionESProducer.algorithm = "AK5CaloHLT"


# customization for CMSSW_5_3_X
if cmsswVersion.startswith('CMSSW_5_3_'):

    # do not override the calo jet energy corrections in 5.3.x for consistency with the current MC samples
    pass


# customization for CMSSW_6_1_X
if cmsswVersion.startswith('CMSSW_6_1_'):

    # force the use of the correct calo jet energy corrections
    if 'hltESPL1FastJetCorrectionESProducer' in process.__dict__:
        process.hltESPL1FastJetCorrectionESProducer.algorithm  = "AK5CaloHLT"

    if 'hltESPL2RelativeCorrectionESProducer' in process.__dict__:
        process.hltESPL2RelativeCorrectionESProducer.algorithm = "AK5CaloHLT"

    if 'hltESPL3AbsoluteCorrectionESProducer' in process.__dict__:
        process.hltESPL3AbsoluteCorrectionESProducer.algorithm = "AK5CaloHLT"

    # adapt the HLT menu to the "prototype for Event Interpretation" development
    if 'hltPFPileUp' in process.__dict__:
        # define new PFCandidateFwdPtrProducer module
        process.hltParticleFlowPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
            src = cms.InputTag('hltParticleFlow')
        )
        # add the new module before the hltPFPileUp module
        _sequence = None
        for _sequence in [ _sequence for _sequence in process.__dict__.itervalues() if isinstance(_sequence, cms._ModuleSequenceType)]:
            try:
                _sequence.insert( _sequence.index(process.hltPFPileUp), process.hltParticleFlowPtrs )
            except ValueError:
                pass
        # reconfigure hltPFPileUp and hltPFNoPileUp to use the new module
        process.hltPFPileUp.PFCandidates       = cms.InputTag( "hltParticleFlowPtrs" )
        process.hltPFNoPileUp.bottomCollection = cms.InputTag( "hltParticleFlowPtrs" )

# override the process name
process.setName_('HLTHIon')

# adapt HLT modules to the correct process name
if 'hltTrigReport' in process.__dict__:
    process.hltTrigReport.HLTriggerResults                    = cms.InputTag( 'TriggerResults', '', 'HLTHIon' )

if 'hltPreExpressCosmicsOutputSmart' in process.__dict__:
    process.hltPreExpressCosmicsOutputSmart.TriggerResultsTag = cms.InputTag( 'TriggerResults', '', 'HLTHIon' )

if 'hltPreExpressOutputSmart' in process.__dict__:
    process.hltPreExpressOutputSmart.TriggerResultsTag        = cms.InputTag( 'TriggerResults', '', 'HLTHIon' )

if 'hltPreDQMForHIOutputSmart' in process.__dict__:
    process.hltPreDQMForHIOutputSmart.TriggerResultsTag       = cms.InputTag( 'TriggerResults', '', 'HLTHIon' )

if 'hltPreDQMForPPOutputSmart' in process.__dict__:
    process.hltPreDQMForPPOutputSmart.TriggerResultsTag       = cms.InputTag( 'TriggerResults', '', 'HLTHIon' )

if 'hltPreHLTDQMResultsOutputSmart' in process.__dict__:
    process.hltPreHLTDQMResultsOutputSmart.TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', 'HLTHIon' )

if 'hltPreHLTDQMOutputSmart' in process.__dict__:
    process.hltPreHLTDQMOutputSmart.TriggerResultsTag         = cms.InputTag( 'TriggerResults', '', 'HLTHIon' )

if 'hltPreHLTMONOutputSmart' in process.__dict__:
    process.hltPreHLTMONOutputSmart.TriggerResultsTag         = cms.InputTag( 'TriggerResults', '', 'HLTHIon' )

if 'hltDQMHLTScalers' in process.__dict__:
    process.hltDQMHLTScalers.triggerResults                   = cms.InputTag( 'TriggerResults', '', 'HLTHIon' )
    process.hltDQMHLTScalers.processname                      = 'HLTHIon'

if 'hltDQML1SeedLogicScalers' in process.__dict__:
    process.hltDQML1SeedLogicScalers.processname              = 'HLTHIon'

# limit the number of events to be processed
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)

# enable the TrigReport and TimeReport
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True )
)

# override the GlobalTag, connection string and pfnPrefix
if 'GlobalTag' in process.__dict__:
    process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
    process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
    from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
    process.GlobalTag = customiseGlobalTag(process.GlobalTag, globaltag = 'auto:hltonline_HIon', conditions = 'L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('L1GtTrigReport')
    process.MessageLogger.categories.append('HLTrigReport')

