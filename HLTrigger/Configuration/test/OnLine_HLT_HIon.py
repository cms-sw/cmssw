# /dev/CMSSW_3_9_0/HIon/V3 (CMSSW_3_8_1_HLT25)

import FWCore.ParameterSet.Config as cms

process = cms.Process( "HLT" )

process.HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_3_9_0/HIon/V3')
)

process.options = cms.untracked.PSet(  Rethrow = cms.untracked.vstring( 'ProductNotFound',
  'TooManyProducts',
  'TooFewProducts' ) )
process.streams = cms.PSet( 
  A = cms.vstring( 'BTau',
    'Commissioning',
    'Cosmics',
    'EGMonitor',
    'Electron',
    'HIAllPhysics',
    'HICorePhysics',
    'HcalHPDNoise',
    'HcalNZS',
    'Jet',
    'JetMETTauMonitor',
    'METFwd',
    'MinimumBias',
    'Mu',
    'MuMonitor',
    'MuOnia',
    'MultiJet',
    'Photon' ),
  ALCAP0 = cms.vstring( 'AlCaP0' ),
  ALCAPHISYM = cms.vstring( 'AlCaPhiSymEcal' ),
  Calibration = cms.vstring( 'TestEnables' ),
  CalibrationHI = cms.vstring( 'EcalHcalCalibrationHI' ),
  DQM = cms.vstring( 'OnlineMonitor',
    'OnlineMonitorHI' ),
  EcalCalibration = cms.vstring( 'EcalLaser' ),
  Express = cms.vstring( 'ExpressPhysics',
    'HIExpressPhysics' ),
  HLTDQM = cms.vstring( 'OnlineHltMonitor',
    'OnlineHltMonitorHI' ),
  HLTDQMResults = cms.vstring( 'OnlineHltResults' ),
  HLTMON = cms.vstring( 'OfflineMonitor',
    'OfflineMonitorHI' ),
  NanoDST = cms.vstring( 'L1Accept' ),
  OnlineErrors = cms.vstring( 'FEDMonitor',
    'LogMonitor' ),
  RPCMON = cms.vstring( 'RPCMonitor' )
)
process.datasets = cms.PSet( 
  AlCaP0 = cms.vstring(  ),
  AlCaPhiSymEcal = cms.vstring(  ),
  BTau = cms.vstring(  ),
  Commissioning = cms.vstring(  ),
  Cosmics = cms.vstring(  ),
  EGMonitor = cms.vstring(  ),
  EcalHcalCalibrationHI = cms.vstring( 'HLT_EcalCalibration',
    'HLT_HcalCalibration_HI' ),
  EcalLaser = cms.vstring( 'HLT_EcalCalibration' ),
  Electron = cms.vstring(  ),
  ExpressPhysics = cms.vstring(  ),
  FEDMonitor = cms.vstring(  ),
  HIAllPhysics = cms.vstring( 'HLT_HIActivityHF_Coincidence3',
    'HLT_HIActivityHF_Single3',
    'HLT_HIActivityL1A',
    'HLT_HIActivityPixel_SingleTrack',
    'HLT_HIActivityPixels',
    'HLT_HIBptxXOR',
    'HLT_HICentralityVeto',
    'HLT_HIClusterVertexCompatibility',
    'HLT_HIDoublePhoton5_CEP_L1R',
    'HLT_HIJet35U',
    'HLT_HIJet50U',
    'HLT_HIJet75U',
    'HLT_HIJet90U',
    'HLT_HIL1Algo_BptxAND',
    'HLT_HIL1Algo_BptxAND_NotBSC2',
    'HLT_HIL1Algo_HFAny',
    'HLT_HIL1Algo_NotBSC2',
    'HLT_HIL1Algo_Unprotected',
    'HLT_HIL1Algo_ZDC',
    'HLT_HIL1DoubleMuOpen',
    'HLT_HIL1ETT30',
    'HLT_HIL1ETT60',
    'HLT_HIL1SingleMu3',
    'HLT_HIL1SingleMu5',
    'HLT_HIL1SingleMu7',
    'HLT_HIL1Tech_BSC_HighMultiplicity',
    'HLT_HIL1Tech_BSC_minBias',
    'HLT_HIL1Tech_BSC_minBias_OR',
    'HLT_HIMinBiasBSC',
    'HLT_HIMinBiasBSC_OR',
    'HLT_HIMinBiasHF',
    'HLT_HIMinBiasPixel_SingleTrack',
    'HLT_HIMinBiasZDC_Calo',
    'HLT_HIPhoton15',
    'HLT_HIPhoton20',
    'HLT_HIPhoton30',
    'HLT_HIRandom',
    'HLT_HIStoppedHSCP35',
    'HLT_HIUpcEcal',
    'HLT_HIUpcMu',
    'HLT_HIZeroBias',
    'HLT_HIZeroBiasPixel_SingleTrack',
    'HLT_HIZeroBiasXOR' ),
  HICorePhysics = cms.vstring( 'HLT_HIJet35U_Core',
    'HLT_HIJet50U_Core',
    'HLT_HIJet75U_Core',
    'HLT_HIJet90U_Core',
    'HLT_HIL1DoubleMuOpen_Core',
    'HLT_HIL1ETT30_Core',
    'HLT_HIL1ETT60_Core',
    'HLT_HIL1SingleMu3_Core',
    'HLT_HIL1SingleMu5_Core',
    'HLT_HIL1SingleMu7_Core',
    'HLT_HIMinBiasBSC_Core',
    'HLT_HIMinBiasBSC_OR_Core',
    'HLT_HIMinBiasHF_Core',
    'HLT_HIPhoton15_Core',
    'HLT_HIPhoton20_Core',
    'HLT_HIPhoton30_Core',
    'HLT_HIUpcEcal_Core',
    'HLT_HIUpcMu_Core' ),
  HIExpressPhysics = cms.vstring( 'HLT_HICentralityVeto',
    'HLT_HIJet35U',
    'HLT_HIL1DoubleMuOpen',
    'HLT_HIL1Tech_BSC_HighMultiplicity',
    'HLT_HIMinBiasBSC_OR',
    'HLT_HIMinBiasHF',
    'HLT_HIPhoton15' ),
  HcalHPDNoise = cms.vstring(  ),
  HcalNZS = cms.vstring(  ),
  Jet = cms.vstring(  ),
  JetMETTauMonitor = cms.vstring(  ),
  L1Accept = cms.vstring( 'HLTriggerFinalPath' ),
  LogMonitor = cms.vstring( 'HLT_LogMonitor' ),
  METFwd = cms.vstring(  ),
  MinimumBias = cms.vstring(  ),
  Mu = cms.vstring(  ),
  MuMonitor = cms.vstring(  ),
  MuOnia = cms.vstring(  ),
  MultiJet = cms.vstring(  ),
  OfflineMonitor = cms.vstring( 'HLT_LogMonitor' ),
  OfflineMonitorHI = cms.vstring( 'HLT_HIActivityHF_Coincidence3',
    'HLT_HIActivityHF_Single3',
    'HLT_HIActivityL1A',
    'HLT_HIActivityPixel_SingleTrack',
    'HLT_HIActivityPixels',
    'HLT_HIBptxXOR',
    'HLT_HICentralityVeto',
    'HLT_HIClusterVertexCompatibility',
    'HLT_HIDoublePhoton5_CEP_L1R',
    'HLT_HIJet35U',
    'HLT_HIJet50U',
    'HLT_HIJet75U',
    'HLT_HIJet90U',
    'HLT_HIL1Algo_BptxAND',
    'HLT_HIL1Algo_BptxAND_NotBSC2',
    'HLT_HIL1Algo_HFAny',
    'HLT_HIL1Algo_NotBSC2',
    'HLT_HIL1Algo_Unprotected',
    'HLT_HIL1Algo_ZDC',
    'HLT_HIL1DoubleMuOpen',
    'HLT_HIL1ETT30',
    'HLT_HIL1ETT60',
    'HLT_HIL1SingleMu3',
    'HLT_HIL1SingleMu5',
    'HLT_HIL1SingleMu7',
    'HLT_HIL1Tech_BSC_HighMultiplicity',
    'HLT_HIL1Tech_BSC_minBias',
    'HLT_HIL1Tech_BSC_minBias_OR',
    'HLT_HIMinBiasBSC_OR',
    'HLT_HIMinBiasHF',
    'HLT_HIMinBiasPixel_SingleTrack',
    'HLT_HIMinBiasZDC_Calo',
    'HLT_HIPhoton15',
    'HLT_HIPhoton20',
    'HLT_HIPhoton30',
    'HLT_HIRandom',
    'HLT_HIStoppedHSCP35',
    'HLT_HIUpcEcal',
    'HLT_HIUpcMu',
    'HLT_HIZeroBias',
    'HLT_HIZeroBiasPixel_SingleTrack' ),
  OnlineHltMonitor = cms.vstring( 'HLT_LogMonitor' ),
  OnlineHltMonitorHI = cms.vstring( 'HLT_HIActivityHF_Coincidence3',
    'HLT_HIActivityHF_Single3',
    'HLT_HIActivityL1A',
    'HLT_HIActivityPixel_SingleTrack',
    'HLT_HIActivityPixels',
    'HLT_HIBptxXOR',
    'HLT_HICentralityVeto',
    'HLT_HIClusterVertexCompatibility',
    'HLT_HIDoublePhoton5_CEP_L1R',
    'HLT_HIJet35U',
    'HLT_HIJet50U',
    'HLT_HIJet75U',
    'HLT_HIJet90U',
    'HLT_HIL1Algo_BptxAND',
    'HLT_HIL1Algo_BptxAND_NotBSC2',
    'HLT_HIL1Algo_HFAny',
    'HLT_HIL1Algo_NotBSC2',
    'HLT_HIL1Algo_Unprotected',
    'HLT_HIL1Algo_ZDC',
    'HLT_HIL1DoubleMuOpen',
    'HLT_HIL1ETT30',
    'HLT_HIL1ETT60',
    'HLT_HIL1SingleMu3',
    'HLT_HIL1SingleMu5',
    'HLT_HIL1SingleMu7',
    'HLT_HIL1Tech_BSC_HighMultiplicity',
    'HLT_HIL1Tech_BSC_minBias',
    'HLT_HIL1Tech_BSC_minBias_OR',
    'HLT_HIMinBiasBSC_OR',
    'HLT_HIMinBiasHF',
    'HLT_HIMinBiasPixel_SingleTrack',
    'HLT_HIMinBiasZDC_Calo',
    'HLT_HIPhoton15',
    'HLT_HIPhoton20',
    'HLT_HIPhoton30',
    'HLT_HIRandom',
    'HLT_HIStoppedHSCP35',
    'HLT_HIUpcEcal',
    'HLT_HIUpcMu',
    'HLT_HIZeroBias',
    'HLT_HIZeroBiasPixel_SingleTrack' ),
  OnlineHltResults = cms.vstring( 'HLTriggerFinalPath' ),
  OnlineMonitor = cms.vstring( 'HLT_EcalCalibration',
    'HLT_LogMonitor' ),
  OnlineMonitorHI = cms.vstring( 'HLT_HIActivityL1A',
    'HLT_HICentralityVeto',
    'HLT_HIJet35U_Core',
    'HLT_HIL1DoubleMuOpen_Core',
    'HLT_HIL1SingleMu3',
    'HLT_HIL1Tech_BSC_HighMultiplicity',
    'HLT_HIMinBiasBSC',
    'HLT_HIMinBiasBSC_OR_Core',
    'HLT_HIMinBiasHF',
    'HLT_HIPhoton15_Core',
    'HLT_HcalCalibration_HI' ),
  Photon = cms.vstring(  ),
  RPCMonitor = cms.vstring(  ),
  TestEnables = cms.vstring(  )
)

process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring( 'file:RelVal_DigiL1Raw_HIon.root' )
)

process.BTagRecord = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "JetTagComputerRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    appendToDataLabel = cms.string( "" ),
    firstValid = cms.vuint32( 1 )
)
process.GlobalTag = cms.ESSource( "PoolDBESSource",
    appendToDataLabel = cms.string( "" ),
    timetype = cms.string( "runnumber" ),
    connect = cms.string( "frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG" ),
    DumpStat = cms.untracked.bool( False ),
    BlobStreamerName = cms.untracked.string( "TBufferBlobStreamingService" ),
    globaltag = cms.string( "GR10_H_V12::All" ),
    DBParameters = cms.PSet( 
      authenticationPath = cms.untracked.string( "." ),
      connectionRetrialPeriod = cms.untracked.int32( 10 ),
      idleConnectionCleanupPeriod = cms.untracked.int32( 10 ),
      messageLevel = cms.untracked.int32( 0 ),
      enablePoolAutomaticCleanUp = cms.untracked.bool( False ),
      enableConnectionSharing = cms.untracked.bool( True ),
      enableReadOnlySessionOnUpdateConnection = cms.untracked.bool( False ),
      connectionTimeOut = cms.untracked.int32( 0 ),
      connectionRetrialTimeOut = cms.untracked.int32( 60 )
    ),
    toGet = cms.VPSet( 
    ),
    RefreshEachRun = cms.untracked.bool( True )
)
process.HepPDTESSource = cms.ESSource( "HepPDTESSource",
    pdtFileName = cms.FileInPath( "SimGeneral/HepPDTESSource/data/pythiaparticle.tbl" ),
    appendToDataLabel = cms.string( "" )
)
process.L2RelativeCorrectionService = cms.ESSource( "LXXXCorrectionService",
    appendToDataLabel = cms.string( "" ),
    level = cms.string( "L2Relative" ),
    algorithm = cms.string( "IC5Calo" ),
    section = cms.string( "" ),
    era = cms.string( "Summer09_7TeV_ReReco332" )
)
process.L3AbsoluteCorrectionService = cms.ESSource( "LXXXCorrectionService",
    appendToDataLabel = cms.string( "" ),
    level = cms.string( "L3Absolute" ),
    algorithm = cms.string( "IC5Calo" ),
    section = cms.string( "" ),
    era = cms.string( "Summer09_7TeV_ReReco332" )
)
process.MCJetCorrectorIcone5 = cms.ESSource( "JetCorrectionServiceChain",
    appendToDataLabel = cms.string( "" ),
    correctors = cms.vstring( 'L2RelativeCorrectionService',
      'L3AbsoluteCorrectionService' ),
    label = cms.string( "MCJetCorrectorIcone5" )
)
process.MCJetCorrectorIcone5HF07 = cms.ESSource( "LXXXCorrectionService",
    appendToDataLabel = cms.string( "" ),
    level = cms.string( "L2Relative" ),
    algorithm = cms.string( "" ),
    section = cms.string( "" ),
    era = cms.string( "HLT" )
)
process.MCJetCorrectorIcone5Unit = cms.ESSource( "LXXXCorrectionService",
    appendToDataLabel = cms.string( "" ),
    level = cms.string( "L2RelativeFlat" ),
    algorithm = cms.string( "" ),
    section = cms.string( "" ),
    era = cms.string( "HLT" )
)
process.eegeom = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "EcalMappingRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    appendToDataLabel = cms.string( "" ),
    firstValid = cms.vuint32( 1 )
)
process.es_hardcode = cms.ESSource( "HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring( 'GainWidths' ),
    appendToDataLabel = cms.string( "" )
)
process.essourceSev = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "HcalSeverityLevelComputerRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    appendToDataLabel = cms.string( "" ),
    firstValid = cms.vuint32( 1 )
)
process.magfield = cms.ESSource( "XMLIdealGeometryESSource",
    rootNodeName = cms.string( "cmsMagneticField:MAGF" ),
    appendToDataLabel = cms.string( "" ),
    geomXMLFiles = cms.vstring( 'Geometry/CMSCommonData/data/normal/cmsextent.xml',
      'Geometry/CMSCommonData/data/cms.xml',
      'Geometry/CMSCommonData/data/cmsMagneticField.xml',
      'MagneticField/GeomBuilder/data/MagneticFieldVolumes_1103l.xml',
      'MagneticField/GeomBuilder/data/MagneticFieldParameters_07_2pi.xml' )
)

process.hltHIPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltHIPixelLayerPairs" ),
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
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  )
)
process.hltHIPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltHIPixelLayerTriplets" ),
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltHISiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  )
)
process.AnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  ComponentName = cms.string( "AnalyticalPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  MaxDPhi = cms.double( 1.6 ),
  appendToDataLabel = cms.string( "" )
)
process.AnyDirectionAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  ComponentName = cms.string( "AnyDirectionAnalyticalPropagator" ),
  PropagationDirection = cms.string( "anyDirection" ),
  MaxDPhi = cms.double( 1.6 ),
  appendToDataLabel = cms.string( "" )
)
process.AutoMagneticFieldESProducer = cms.ESProducer( "AutoMagneticFieldESProducer",
  label = cms.untracked.string( "" ),
  valueOverride = cms.int32( -1 ),
  appendToDataLabel = cms.string( "" ),
  nominalCurrents = cms.untracked.vint32( -1, 0, 9558, 14416, 16819, 18268, 19262 ),
  mapLabels = cms.untracked.vstring( '090322_3_8t',
    '0t',
    '071212_2t',
    '071212_3t',
    '071212_3_5t',
    '090322_3_8t',
    '071212_4t' )
)
process.CSCGeometryESModule = cms.ESProducer( "CSCGeometryESModule",
  alignmentsLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  useRealWireGeometry = cms.bool( True ),
  useOnlyWiresInME1a = cms.bool( False ),
  useGangedStripsInME1a = cms.bool( True ),
  useCentreTIOffsets = cms.bool( False ),
  useDDD = cms.bool( False ),
  applyAlignment = cms.bool( True )
)
process.CaloGeometryBuilder = cms.ESProducer( "CaloGeometryBuilder",
  appendToDataLabel = cms.string( "" ),
  SelectedCalos = cms.vstring( 'HCAL',
    'ZDC',
    'EcalBarrel',
    'EcalEndcap',
    'EcalPreshower',
    'TOWER' )
)
process.CaloTopologyBuilder = cms.ESProducer( "CaloTopologyBuilder",
  appendToDataLabel = cms.string( "" )
)
process.CaloTowerConstituentsMapBuilder = cms.ESProducer( "CaloTowerConstituentsMapBuilder",
  MapFile = cms.untracked.string( "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz" ),
  appendToDataLabel = cms.string( "" )
)
process.CaloTowerGeometryFromDBEP = cms.ESProducer( "CaloTowerGeometryFromDBEP",
  appendToDataLabel = cms.string( "" ),
  applyAlignment = cms.bool( False )
)
process.Chi2EstimatorForRefit = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "Chi2EstimatorForRefit" ),
  MaxChi2 = cms.double( 100000.0 ),
  nSigma = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
process.Chi2MeasurementEstimator = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "Chi2" ),
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
process.DTGeometryESModule = cms.ESProducer( "DTGeometryESModule",
  alignmentsLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  fromDDD = cms.bool( False ),
  applyAlignment = cms.bool( True )
)
process.DummyDetLayerGeometry = cms.ESProducer( "DetLayerGeometryESProducer",
  ComponentName = cms.string( "DummyDetLayerGeometry" ),
  appendToDataLabel = cms.string( "" )
)
process.ESUnpackerWorkerESProducer = cms.ESProducer( "ESUnpackerWorkerESProducer",
  ComponentName = cms.string( "esRawToRecHit" ),
  appendToDataLabel = cms.string( "" ),
  DCCDataUnpacker = cms.PSet(  LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ) ),
  RHAlgo = cms.PSet( 
    Type = cms.string( "ESRecHitWorker" ),
    ESGain = cms.int32( 2 ),
    ESMIPkeV = cms.double( 81.08 ),
    ESMIPADC = cms.double( 55.0 ),
    ESBaseline = cms.int32( 0 ),
    ESRecoAlgo = cms.int32( 0 )
  )
)
process.EcalBarrelGeometryFromDBEP = cms.ESProducer( "EcalBarrelGeometryFromDBEP",
  appendToDataLabel = cms.string( "" ),
  applyAlignment = cms.bool( True )
)
process.EcalElectronicsMappingBuilder = cms.ESProducer( "EcalElectronicsMappingBuilder",
  appendToDataLabel = cms.string( "" )
)
process.EcalEndcapGeometryFromDBEP = cms.ESProducer( "EcalEndcapGeometryFromDBEP",
  appendToDataLabel = cms.string( "" ),
  applyAlignment = cms.bool( True )
)
process.EcalLaserCorrectionService = cms.ESProducer( "EcalLaserCorrectionService",
  appendToDataLabel = cms.string( "" )
)
process.EcalPreshowerGeometryFromDBEP = cms.ESProducer( "EcalPreshowerGeometryFromDBEP",
  appendToDataLabel = cms.string( "" ),
  applyAlignment = cms.bool( True )
)
process.EcalRegionCablingESProducer = cms.ESProducer( "EcalRegionCablingESProducer",
  appendToDataLabel = cms.string( "" ),
  esMapping = cms.PSet(  LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ) )
)
process.EcalTrigTowerConstituentsMapBuilder = cms.ESProducer( "EcalTrigTowerConstituentsMapBuilder",
  MapFile = cms.untracked.string( "Geometry/EcalMapping/data/EndCap_TTMap.txt" ),
  appendToDataLabel = cms.string( "" )
)
process.EcalUnpackerWorkerESProducer = cms.ESProducer( "EcalUnpackerWorkerESProducer",
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
    flagsMapDBReco = cms.vint32( 0, 0, 0, 0, 4, -1, -1, -1, 4, 4, 6, 6, 6, 7, 8 ),
    Type = cms.string( "EcalRecHitWorkerSimple" ),
    killDeadChannels = cms.bool( True ),
    ChannelStatusToBeExcluded = cms.vint32( 10, 11, 12, 13, 14, 78, 142 ),
    laserCorrection = cms.bool( False )
  )
)
process.FastSteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "FastSteppingHelixPropagatorAny" ),
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
process.FastSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "FastSteppingHelixPropagatorOpposite" ),
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
process.FitterRK = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "FitterRK" ),
  Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  RecoGeometry = cms.string( "DummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.FittingSmootherRK = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "FittingSmootherRK" ),
  Fitter = cms.string( "FitterRK" ),
  Smoother = cms.string( "SmootherRK" ),
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinNumberOfHits = cms.int32( 5 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.GlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer",
  appendToDataLabel = cms.string( "" )
)
process.HITTRHBuilderWithoutRefit = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "HITTRHBuilderWithoutRefit" ),
  StripCPE = cms.string( "Fake" ),
  PixelCPE = cms.string( "Fake" ),
  Matcher = cms.string( "Fake" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.HcalGeometryFromDBEP = cms.ESProducer( "HcalGeometryFromDBEP",
  appendToDataLabel = cms.string( "" ),
  applyAlignment = cms.bool( False )
)
process.HcalTopologyIdealEP = cms.ESProducer( "HcalTopologyIdealEP",
  appendToDataLabel = cms.string( "" )
)
process.KFFitterSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "KFFitterSmootherForL2Muon" ),
  Fitter = cms.string( "KFTrajectoryFitterForL2Muon" ),
  Smoother = cms.string( "KFTrajectorySmootherForL2Muon" ),
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinNumberOfHits = cms.int32( 5 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.KFSmootherForMuonTrackLoader = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "KFSmootherForMuonTrackLoader" ),
  Propagator = cms.string( "SmartPropagatorAnyOpposite" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  RecoGeometry = cms.string( "DummyDetLayerGeometry" ),
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.KFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "KFTrajectoryFitterForL2Muon" ),
  Propagator = cms.string( "FastSteppingHelixPropagatorAny" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  RecoGeometry = cms.string( "DummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.KFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "KFTrajectorySmootherForL2Muon" ),
  Propagator = cms.string( "FastSteppingHelixPropagatorOpposite" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  RecoGeometry = cms.string( "DummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.KFUpdatorESProducer = cms.ESProducer( "KFUpdatorESProducer",
  ComponentName = cms.string( "KFUpdator" ),
  appendToDataLabel = cms.string( "" )
)
process.L1GtTriggerMaskAlgoTrigTrivialProducer = cms.ESProducer( "L1GtTriggerMaskAlgoTrigTrivialProducer",
  appendToDataLabel = cms.string( "" ),
  TriggerMask = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
)
process.L1GtTriggerMaskTechTrigTrivialProducer = cms.ESProducer( "L1GtTriggerMaskTechTrigTrivialProducer",
  appendToDataLabel = cms.string( "" ),
  TriggerMask = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
)
process.L3MuKFFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "L3MuKFFitter" ),
  Propagator = cms.string( "SmartPropagatorAny" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  RecoGeometry = cms.string( "DummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.MaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "PropagatorWithMaterial" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Mass = cms.double( 0.105 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" )
)
process.hltMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  ComponentName = cms.string( "hltMeasurementTracker" ),
  PixelCPE = cms.string( "PixelCPEGeneric" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  HitMatcher = cms.string( "StandardMatcher" ),
  Regional = cms.bool( True ),
  OnDemand = cms.bool( True ),
  UsePixelModuleQualityDB = cms.bool( True ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripModuleQualityDB = cms.bool( True ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  SiStripQualityLabel = cms.string( "" ),
  switchOffPixelsIfEmpty = cms.bool( True ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  stripClusterProducer = cms.string( "hltSiStripClusters" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
  appendToDataLabel = cms.string( "" ),
  inactivePixelDetectorLabels = cms.VInputTag(  ),
  inactiveStripDetectorLabels = cms.VInputTag(  ),
  badStripCuts = cms.PSet( 
    TIB = cms.PSet( 
      maxBad = cms.uint32( 9999 ),
      maxConsecutiveBad = cms.uint32( 9999 )
    ),
    TOB = cms.PSet( 
      maxBad = cms.uint32( 9999 ),
      maxConsecutiveBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxBad = cms.uint32( 9999 ),
      maxConsecutiveBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxBad = cms.uint32( 9999 ),
      maxConsecutiveBad = cms.uint32( 9999 )
    )
  )
)
process.MuonCkfTrajectoryBuilder = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "muonCkfTrajectoryBuilder" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltMeasurementTracker" ),
  trajectoryFilterName = cms.string( "muonCkfTrajectoryFilter" ),
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
process.MuonDetLayerGeometryESProducer = cms.ESProducer( "MuonDetLayerGeometryESProducer",
  appendToDataLabel = cms.string( "" )
)
process.MuonNumberingInitialization = cms.ESProducer( "MuonNumberingInitialization",
  appendToDataLabel = cms.string( "" )
)
process.MuonTransientTrackingRecHitBuilderESProducer = cms.ESProducer( "MuonTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "MuonRecHitBuilder" ),
  appendToDataLabel = cms.string( "" )
)
process.OppositeMaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "PropagatorWithMaterialOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Mass = cms.double( 0.105 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" )
)
process.PixelCPEGenericESProducer = cms.ESProducer( "PixelCPEGenericESProducer",
  ComponentName = cms.string( "PixelCPEGeneric" ),
  eff_charge_cut_lowX = cms.double( 0.0 ),
  eff_charge_cut_lowY = cms.double( 0.0 ),
  eff_charge_cut_highX = cms.double( 1.0 ),
  eff_charge_cut_highY = cms.double( 1.0 ),
  size_cutX = cms.double( 3.0 ),
  size_cutY = cms.double( 3.0 ),
  EdgeClusterErrorX = cms.double( 50.0 ),
  EdgeClusterErrorY = cms.double( 85.0 ),
  inflate_errors = cms.bool( False ),
  inflate_all_errors_no_trk_angle = cms.bool( False ),
  UseErrorsFromTemplates = cms.bool( True ),
  TruncatePixelCharge = cms.bool( True ),
  IrradiationBiasCorrection = cms.bool( False ),
  DoCosmics = cms.bool( False ),
  LoadTemplatesFromDB = cms.bool( True ),
  appendToDataLabel = cms.string( "" ),
  TanLorentzAnglePerTesla = cms.double( 0.106 ),
  PixelErrorParametrization = cms.string( "NOTcmsim" ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 )
)
process.RPCGeometryESModule = cms.ESProducer( "RPCGeometryESModule",
  useDDD = cms.untracked.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.RungeKuttaTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "RungeKuttaTrackerPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Mass = cms.double( 0.105 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( True ),
  ptMin = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" )
)
process.SiStripGainESProducer = cms.ESProducer( "SiStripGainESProducer",
  AutomaticNormalization = cms.bool( False ),
  NormalizationFactor = cms.double( 1.0 ),
  printDebug = cms.untracked.bool( False ),
  APVGain = cms.VPSet( 
    cms.PSet(  Record = cms.string( "SiStripApvGainRcd" ),
      Label = cms.untracked.string( "" ),
      NormalizationFactor = cms.untracked.double( 1.0 )
    ),
    cms.PSet(  Record = cms.string( "SiStripApvGain2Rcd" ),
      Label = cms.untracked.string( "" ),
      NormalizationFactor = cms.untracked.double( 1.0 )
    )
  )
)
process.SiStripQualityESProducer = cms.ESProducer( "SiStripQualityESProducer",
  appendToDataLabel = cms.string( "" ),
  PrintDebugOutput = cms.bool( False ),
  ThresholdForReducedGranularity = cms.double( 0.3 ),
  UseEmptyRunInfo = cms.bool( False ),
  ReduceGranularity = cms.bool( False ),
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
    ),
    cms.PSet(  record = cms.string( "RunInfoRcd" ),
      tag = cms.string( "" )
    )
  )
)
process.SiStripRecHitMatcherESProducer = cms.ESProducer( "SiStripRecHitMatcherESProducer",
  ComponentName = cms.string( "StandardMatcher" ),
  NSigmaInside = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
process.SiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 )
)
process.SlaveField0 = cms.ESProducer( "UniformMagneticFieldESProducer",
  ZFieldInTesla = cms.double( 0.0 ),
  label = cms.untracked.string( "slave_0" ),
  appendToDataLabel = cms.string( "" )
)
process.SlaveField20 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  label = cms.untracked.string( "slave_20" ),
  version = cms.string( "OAE_1103l_071212" ),
  appendToDataLabel = cms.string( "" ),
  parameters = cms.PSet(  BValue = cms.string( "2_0T" ) )
)
process.SlaveField30 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  label = cms.untracked.string( "slave_30" ),
  version = cms.string( "OAE_1103l_071212" ),
  appendToDataLabel = cms.string( "" ),
  parameters = cms.PSet(  BValue = cms.string( "3_0T" ) )
)
process.SlaveField35 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  label = cms.untracked.string( "slave_35" ),
  version = cms.string( "OAE_1103l_071212" ),
  appendToDataLabel = cms.string( "" ),
  parameters = cms.PSet(  BValue = cms.string( "3_5T" ) )
)
process.SlaveField38 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  label = cms.untracked.string( "slave_38" ),
  version = cms.string( "OAE_1103l_071212" ),
  appendToDataLabel = cms.string( "" ),
  parameters = cms.PSet(  BValue = cms.string( "3_8T" ) )
)
process.SlaveField40 = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  label = cms.untracked.string( "slave_40" ),
  version = cms.string( "OAE_1103l_071212" ),
  appendToDataLabel = cms.string( "" ),
  parameters = cms.PSet(  BValue = cms.string( "4_0T" ) )
)
process.SmartPropagator = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "SmartPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAlong" ),
  appendToDataLabel = cms.string( "" )
)
process.SmartPropagatorAny = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "SmartPropagatorAny" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  appendToDataLabel = cms.string( "" )
)
process.SmartPropagatorAnyOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "SmartPropagatorAnyOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  appendToDataLabel = cms.string( "" )
)
process.SmartPropagatorOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "SmartPropagatorOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorOpposite" ),
  appendToDataLabel = cms.string( "" )
)
process.SmootherRK = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "SmootherRK" ),
  Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  RecoGeometry = cms.string( "DummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.SteppingHelixPropagatorAlong = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "SteppingHelixPropagatorAlong" ),
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
process.SteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
process.SteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "SteppingHelixPropagatorOpposite" ),
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
process.StraightLinePropagator = cms.ESProducer( "StraightLinePropagatorESProducer",
  ComponentName = cms.string( "StraightLinePropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  appendToDataLabel = cms.string( "" )
)
process.StripCPEfromTrackAngleESProducer = cms.ESProducer( "StripCPEESProducer",
  ComponentName = cms.string( "StripCPEfromTrackAngle" ),
  appendToDataLabel = cms.string( "" ),
  TanDiffusionAngle = cms.double( 0.01 ),
  ThicknessRelativeUncertainty = cms.double( 0.02 ),
  NoiseThreshold = cms.double( 2.3 ),
  MaybeNoiseThreshold = cms.double( 3.5 ),
  UncertaintyScaling = cms.double( 1.42 ),
  MinimumUncertainty = cms.double( 0.01 )
)
process.TTRHBuilderPixelOnly = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "TTRHBuilderPixelOnly" ),
  StripCPE = cms.string( "Fake" ),
  PixelCPE = cms.string( "PixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.TrackerDigiGeometryESModule = cms.ESProducer( "TrackerDigiGeometryESModule",
  alignmentsLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  applyAlignment = cms.bool( True ),
  fromDDD = cms.bool( False )
)
process.TrackerGeometricDetESModule = cms.ESProducer( "TrackerGeometricDetESModule",
  fromDDD = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.TrackerRecoGeometryESProducer = cms.ESProducer( "TrackerRecoGeometryESProducer",
  appendToDataLabel = cms.string( "" )
)
process.TransientTrackBuilderESProducer = cms.ESProducer( "TransientTrackBuilderESProducer",
  ComponentName = cms.string( "TransientTrackBuilder" ),
  appendToDataLabel = cms.string( "" )
)
process.VBF0 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  label = cms.untracked.string( "0t" ),
  version = cms.string( "grid_1103l_071212_2t" ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  paramLabel = cms.string( "slave_0" ),
  appendToDataLabel = cms.string( "" ),
  scalingVolumes = cms.vint32(  ),
  scalingFactors = cms.vdouble(  ),
  findVolumeTolerance = cms.double( 0.0 ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.VBF20 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  label = cms.untracked.string( "071212_2t" ),
  version = cms.string( "grid_1103l_071212_2t" ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  paramLabel = cms.string( "slave_20" ),
  appendToDataLabel = cms.string( "" ),
  scalingVolumes = cms.vint32(  ),
  scalingFactors = cms.vdouble(  ),
  findVolumeTolerance = cms.double( 0.0 ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.VBF30 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  label = cms.untracked.string( "071212_3t" ),
  version = cms.string( "grid_1103l_071212_3t" ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  paramLabel = cms.string( "slave_30" ),
  appendToDataLabel = cms.string( "" ),
  scalingVolumes = cms.vint32(  ),
  scalingFactors = cms.vdouble(  ),
  findVolumeTolerance = cms.double( 0.0 ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.VBF35 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  label = cms.untracked.string( "071212_3_5t" ),
  version = cms.string( "grid_1103l_071212_3_5t" ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  paramLabel = cms.string( "slave_35" ),
  appendToDataLabel = cms.string( "" ),
  scalingVolumes = cms.vint32(  ),
  scalingFactors = cms.vdouble(  ),
  findVolumeTolerance = cms.double( 0.0 ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.VBF38 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  label = cms.untracked.string( "090322_3_8t" ),
  version = cms.string( "grid_1103l_090322_3_8t" ),
  overrideMasterSector = cms.bool( False ),
  useParametrizedTrackerField = cms.bool( True ),
  paramLabel = cms.string( "slave_38" ),
  appendToDataLabel = cms.string( "" ),
  scalingVolumes = cms.vint32( 14100, 14200, 17600, 17800, 17900, 18100, 18300, 18400, 18600, 23100, 23300, 23400, 23600, 23800, 23900, 24100, 28600, 28800, 28900, 29100, 29300, 29400, 29600, 28609, 28809, 28909, 29109, 29309, 29409, 29609, 28610, 28810, 28910, 29110, 29310, 29410, 29610, 28611, 28811, 28911, 29111, 29311, 29411, 29611 ),
  scalingFactors = cms.vdouble( 1.0, 1.0, 0.994, 1.004, 1.004, 1.005, 1.004, 1.004, 0.994, 0.965, 0.958, 0.958, 0.953, 0.958, 0.958, 0.965, 0.918, 0.924, 0.924, 0.906, 0.924, 0.924, 0.918, 0.991, 0.998, 0.998, 0.978, 0.998, 0.998, 0.991, 0.991, 0.998, 0.998, 0.978, 0.998, 0.998, 0.991, 0.991, 0.998, 0.998, 0.978, 0.998, 0.998, 0.991 ),
  findVolumeTolerance = cms.double( 0.0 ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.VBF40 = cms.ESProducer( "VolumeBasedMagneticFieldESProducer",
  label = cms.untracked.string( "071212_4t" ),
  version = cms.string( "grid_1103l_071212_4t" ),
  overrideMasterSector = cms.bool( True ),
  useParametrizedTrackerField = cms.bool( True ),
  paramLabel = cms.string( "slave_40" ),
  appendToDataLabel = cms.string( "" ),
  scalingVolumes = cms.vint32(  ),
  scalingFactors = cms.vdouble(  ),
  findVolumeTolerance = cms.double( 0.0 ),
  cacheLastVolume = cms.untracked.bool( True )
)
process.XMLFromDBSource = cms.ESProducer( "XMLIdealGeometryESProducer",
  rootDDName = cms.string( "cms:OCMS" ),
  label = cms.string( "Extended" ),
  appendToDataLabel = cms.string( "" )
)
process.ZdcGeometryFromDBEP = cms.ESProducer( "ZdcGeometryFromDBEP",
  appendToDataLabel = cms.string( "" ),
  applyAlignment = cms.bool( False )
)
process.bJetRegionalTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "bJetRegionalTrajectoryBuilder" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltMeasurementTracker" ),
  trajectoryFilterName = cms.string( "bJetRegionalTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.bJetRegionalTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "bJetRegionalTrajectoryFilter" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minimumNumberOfHits = cms.int32( 5 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 8 ),
    maxConsecLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 1.0 )
  )
)
process.caloDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "CaloDetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.ckfBaseTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "ckfBaseTrajectoryFilter" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minimumNumberOfHits = cms.int32( 5 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 0.9 )
  )
)
process.ecalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "EcalDetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.02 ),
  nEta = cms.int32( 300 ),
  nPhi = cms.int32( 360 ),
  includeBadChambers = cms.bool( False )
)
process.hcalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HcalDetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.hcalRecAlgos = cms.ESProducer( "HcalRecAlgoESProducer",
  SeverityLevels = cms.VPSet( 
    cms.PSet(  RecHitFlags = cms.vstring( '' ),
      ChannelStatus = cms.vstring( '' ),
      Level = cms.int32( 0 )
    ),
    cms.PSet(  Level = cms.int32( 1 ),
      RecHitFlags = cms.vstring( '' ),
      ChannelStatus = cms.vstring( 'HcalCellCaloTowerProb' )
    ),
    cms.PSet(  Level = cms.int32( 5 ),
      RecHitFlags = cms.vstring( 'HSCP_R1R2',
        'HSCP_FracLeader',
        'HSCP_OuterEnergy',
        'HSCP_ExpFit',
        'ADCSaturationBit' ),
      ChannelStatus = cms.vstring( '' )
    ),
    cms.PSet(  Level = cms.int32( 8 ),
      RecHitFlags = cms.vstring( 'HBHEHpdHitMultiplicity',
        'HBHEPulseShape',
        'HOBit',
        'HFDigiTime',
        'HFInTimeWindow',
        'HFS8S1Ratio',
        'ZDCBit',
        'CalibrationBit',
        'TimingErrorBit' ),
      ChannelStatus = cms.vstring( '' )
    ),
    cms.PSet(  Level = cms.int32( 11 ),
      RecHitFlags = cms.vstring( 'HFLongShort' ),
      ChannelStatus = cms.vstring( '' )
    ),
    cms.PSet(  Level = cms.int32( 12 ),
      RecHitFlags = cms.vstring( '' ),
      ChannelStatus = cms.vstring( 'HcalCellCaloTowerMask' )
    ),
    cms.PSet(  Level = cms.int32( 15 ),
      RecHitFlags = cms.vstring( '' ),
      ChannelStatus = cms.vstring( 'HcalCellHot' )
    ),
    cms.PSet(  Level = cms.int32( 20 ),
      RecHitFlags = cms.vstring( '' ),
      ChannelStatus = cms.vstring( 'HcalCellOff',
        'HcalCellDead' )
    )
  ),
  DropChannelStatusBits = cms.vstring( 'HcalCellMask',
    'HcalCellOff',
    'HcalCellDead' ),
  appendToDataLabel = cms.string( "" ),
  RecoveredRecHitBits = cms.vstring( 'TimingAddedBit',
    'TimingSubtractedBit' )
)
process.hcal_db_producer = cms.ESProducer( "HcalDbProducer",
  appendToDataLabel = cms.string( "" )
)
process.hltCkfTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryBuilder" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltMeasurementTracker" ),
  trajectoryFilterName = cms.string( "ckfBaseTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
process.hltCkfTrajectoryBuilderMumu = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryBuilderMumu" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltCkfTrajectoryFilterMumu" ),
  maxCand = cms.int32( 3 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.hltCkfTrajectoryFilterMumu = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryFilterMumu" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minimumNumberOfHits = cms.int32( 5 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 5 ),
    maxConsecLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 3.0 )
  )
)
process.hltKFFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltKFFitter" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  RecoGeometry = cms.string( "DummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltKFFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltKFFittingSmoother" ),
  Fitter = cms.string( "hltKFFitter" ),
  Smoother = cms.string( "hltKFSmoother" ),
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinNumberOfHits = cms.int32( 5 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.hltKFSmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltKFSmoother" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  RecoGeometry = cms.string( "DummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltMixedLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltMixedLayerPairs" ),
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
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet( 
    useRingSlector = cms.bool( True ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  )
)
process.hltMuTrackJpsiTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltMuTrackJpsiTrajectoryBuilder" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltMeasurementTracker" ),
  trajectoryFilterName = cms.string( "hltMuTrackJpsiTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.hltMuTrackJpsiTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltMuTrackJpsiTrajectoryFilter" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minimumNumberOfHits = cms.int32( 5 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 8 ),
    maxConsecLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 1.0 )
  )
)
process.hltPixelLayerPairs = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltPixelLayerPairs" ),
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
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  )
)
process.hltPixelLayerTriplets = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "hltPixelLayerTriplets" ),
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  )
)
process.hoDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HODetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 30 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.muonCkfTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "muonCkfTrajectoryFilter" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    chargeSignificance = cms.double( -1.0 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 0.9 )
  )
)
process.muonDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "MuonDetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.125 ),
  nEta = cms.int32( 48 ),
  nPhi = cms.int32( 48 ),
  includeBadChambers = cms.bool( False )
)
process.navigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "SimpleNavigationSchool" ),
  appendToDataLabel = cms.string( "" )
)
process.pixellayertripletsHITHB = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "PixelLayerTripletsHITHB" ),
  layerList = cms.vstring( 'BPix1+BPix2+BPix3' ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  )
)
process.pixellayertripletsHITHE = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "PixelLayerTripletsHITHE" ),
  layerList = cms.vstring( 'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  )
)
process.preshowerDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "PreshowerDetIdAssociator" ),
  appendToDataLabel = cms.string( "" ),
  etaBinSize = cms.double( 0.1 ),
  nEta = cms.int32( 60 ),
  nPhi = cms.int32( 30 ),
  includeBadChambers = cms.bool( False )
)
process.siPixelTemplateDBObjectESProducer = cms.ESProducer( "SiPixelTemplateDBObjectESProducer",
  appendToDataLabel = cms.string( "" )
)
process.sistripconn = cms.ESProducer( "SiStripConnectivity" )
process.softLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  appendToDataLabel = cms.string( "" ),
  distance = cms.double( 0.5 )
)
process.softLeptonByPt = cms.ESProducer( "LeptonTaggerByPtESProducer",
  appendToDataLabel = cms.string( "" ),
  ipSign = cms.string( "any" )
)
process.trackCounting3D2nd = cms.ESProducer( "TrackCountingESProducer",
  appendToDataLabel = cms.string( "" ),
  nthTrack = cms.int32( 2 ),
  impactParameterType = cms.int32( 0 ),
  deltaR = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 5.0 ),
  maximumDistanceToJetAxis = cms.double( 0.07 ),
  trackQualityClass = cms.string( "any" )
)
process.trajBuilderL3 = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "trajBuilderL3" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltMeasurementTracker" ),
  trajectoryFilterName = cms.string( "trajFilterL3" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.trajFilterL3 = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "trajFilterL3" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    minimumNumberOfHits = cms.int32( 5 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 7 ),
    maxConsecLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 0.9 )
  )
)
process.hltTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  appendToDataLabel = cms.string( "" ),
  fractionShared = cms.double( 0.5 ),
  allowSharedFirstHit = cms.bool( False )
)
process.ttrhbwr = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "WithTrackAngle" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  PixelCPE = cms.string( "PixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)

process.DQM = cms.Service( "DQM",
)
process.DQMStore = cms.Service( "DQMStore",
)
process.DTDataIntegrityTask = cms.Service( "DTDataIntegrityTask",
    getSCInfo = cms.untracked.bool( True ),
    processingMode = cms.untracked.string( "HLT" )
)
process.MessageLogger = cms.Service( "MessageLogger",
    destinations = cms.untracked.vstring( 'warnings',
      'errors',
      'infos',
      'debugs',
      'cout',
      'cerr' ),
    categories = cms.untracked.vstring( 'FwkJob',
      'FwkReport',
      'FwkSummary',
      'Root_NoDictionary' ),
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
    ),
    cout = cms.untracked.PSet( 
      threshold = cms.untracked.string( "ERROR" ),
    ),
    errors = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      placeholder = cms.untracked.bool( True ),
    ),
    warnings = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      placeholder = cms.untracked.bool( True ),
    ),
    infos = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      Root_NoDictionary = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      placeholder = cms.untracked.bool( True ),
    ),
    debugs = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      placeholder = cms.untracked.bool( True ),
    ),
    fwkJobReports = cms.untracked.vstring( 'FrameworkJobReport' ),
    FrameworkJobReport = cms.untracked.PSet( 
      default = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      FwkJob = cms.untracked.PSet(  limit = cms.untracked.int32( 10000000 ) )
    ),
    suppressWarning = cms.untracked.vstring( 'hltOnlineBeamSpot',
      'hltPixelTracksForMinBias',
      'hltPixelTracksForHighMult',
      'hltHITPixelTracksHE',
      'hltHITPixelTracksHB',
      'hltSiPixelClusters',
      'hltPixelTracks' ),
    threshold = cms.untracked.string( "INFO" ),
)
process.MicroStateService = cms.Service( "MicroStateService",
)
process.ModuleWebRegistry = cms.Service( "ModuleWebRegistry",
)
process.PrescaleService = cms.Service( "PrescaleService",
    lvl1DefaultLabel = cms.untracked.string( "2E32" ),
    lvl1Labels = cms.vstring( '4E32',
      '2E32',
      '1.7E32',
      '1.4E32',
      '1.1E32',
      '0.8E32' ),
    prescaleTable = cms.VPSet( 
      cms.PSet(  pathName = cms.string( "HLTDQMResultsOutput" ),
        prescales = cms.vuint32( 10, 10, 10, 10, 10, 10 )
      ),
      cms.PSet(  pathName = cms.string( "HLTMONOutput" ),
        prescales = cms.vuint32( 100, 100, 100, 100, 100, 100 )
      )
    )
)
process.UpdaterService = cms.Service( "UpdaterService",
)

process.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
process.hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
process.hltGtDigis = cms.EDProducer( "L1GlobalTriggerRawToDigi",
    DaqGtInputTag = cms.InputTag( "rawDataCollector" ),
    DaqGtFedId = cms.untracked.int32( 813 ),
    ActiveBoardsMask = cms.uint32( 0xffff ),
    UnpackBxInEvent = cms.int32( 5 ),
    Verbosity = cms.untracked.int32( 0 )
)
process.hltPreLogMonitor = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltLogMonitorFilter = cms.EDFilter( "HLTLogMonitorFilter",
    default_threshold = cms.uint32( 10 ),
    categories = cms.VPSet( 
    )
)
process.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
process.hltCalibrationEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 2 )
)
process.hltPreEcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltEcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 )
)
process.hltTriggerType = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 1 )
)
process.hltGctDigis = cms.EDProducer( "GctRawToDigi",
    inputLabel = cms.InputTag( "rawDataCollector" ),
    gctFedId = cms.untracked.int32( 745 ),
    hltMode = cms.bool( True ),
    numberOfGctSamplesToUnpack = cms.uint32( 1 ),
    numberOfRctSamplesToUnpack = cms.uint32( 1 ),
    unpackSharedRegions = cms.bool( False ),
    unpackerVersion = cms.uint32( 0 )
)
process.hltL1GtObjectMap = cms.EDProducer( "L1GlobalTrigger",
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
    TechnicalTriggersInputTags = cms.VInputTag( 'simBscDigis' ),
    RecordLength = cms.vint32( 3, 0 )
)
process.hltL1extraParticles = cms.EDProducer( "L1ExtraParticlesProd",
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
process.hltScalersRawToDigi = cms.EDProducer( "ScalersRawToDigi",
    scalersInputTag = cms.InputTag( "rawDataCollector" )
)
process.hltOnlineBeamSpot = cms.EDProducer( "BeamSpotOnlineProducer",
    label = cms.InputTag( "hltScalersRawToDigi" ),
    changeToCMSCoordinates = cms.bool( False ),
    maxRadius = cms.double( 2.0 ),
    maxZ = cms.double( 40.0 ),
    setSigmaZ = cms.double( 10.0 ),
    gtEvmLabel = cms.InputTag( "" )
)
process.hltOfflineBeamSpot = cms.EDProducer( "BeamSpotProducer" )
process.hltL1sHIZeroBias = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxPlusANDMinus" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIZeroBiasXORonly = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxXOR" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIZeroBiasXOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIZeroBiasXOR = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxPlusANDMinus OR L1_BptxXOR" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIZeroBiasPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltSiPixelDigis = cms.EDProducer( "SiPixelRawToDigi",
    IncludeErrors = cms.bool( False ),
    UseQualityInfo = cms.bool( False ),
    InputLabel = cms.InputTag( "rawDataCollector" )
)
process.hltHISiPixelClusters = cms.EDProducer( "SiPixelClusterProducer",
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
process.hltHISiPixelRecHits = cms.EDProducer( "SiPixelRecHitConverter",
    src = cms.InputTag( "hltHISiPixelClusters" ),
    CPE = cms.string( "PixelCPEGeneric" )
)
process.hltHIPixelClusterVertices = cms.EDProducer( "HIPixelClusterVtxProducer",
    pixelRecHits = cms.string( "hltHISiPixelRecHits" ),
    minZ = cms.double( -20.0 ),
    maxZ = cms.double( 20.05 ),
    zStep = cms.double( 0.1 )
)
process.hltPixelTracksForHITrackTrigger = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( False ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "HITrackingRegionForPrimaryVtxProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.1 ),
        ptMin = cms.double( 0.7 ),
        directionXCoord = cms.double( 1.0 ),
        directionZCoord = cms.double( 0.0 ),
        directionYCoord = cms.double( 1.0 ),
        useFoundVertices = cms.bool( True ),
        doVariablePtMin = cms.bool( True ),
        nSigmaZ = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
        useFixedError = cms.bool( True ),
        fixedError = cms.double( 2.0 ),
        sigmaZVertex = cms.double( 3.0 ),
        siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" ),
        VertexCollection = cms.InputTag( "hltHIPixelClusterVertices" )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      SeedingLayers = cms.string( "hltHIPixelLayerTriplets" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.06 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.06 ),
        maxElement = cms.uint32( 10000 )
      )
    ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" )
    ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIProtoTrackFilter" ),
      ptMin = cms.double( 1.0 ),
      tipMax = cms.double( 1.0 ),
      doVariablePtMin = cms.bool( True ),
      beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
      siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) )
)
process.hltPixelCandsForHITrackTrigger = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPixelTracksForHITrackTrigger" ),
    particleType = cms.string( "pi+" )
)
process.hltHISinglePixelTrackFilter1 = cms.EDFilter( "HLTPixlMBFilt",
    pixlTag = cms.InputTag( "hltPixelCandsForHITrackTrigger" ),
    MinPt = cms.double( 0.0 ),
    MinTrks = cms.uint32( 1 ),
    MinSep = cms.double( 1.0 )
)
process.hltL1sHIMinBiasBSC = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BscMinBiasThreshold1_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIMinBiasBSC = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIMinBiasBSCCore = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIMinBiasBSCOR = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BscMinBiasOR_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIMinBiasBSCOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIMinBiasBSCORCore = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIMinBiasHF = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HcalHfCoincidencePm_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIMinBiasHF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIMinBiasHFCore = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIMinBiasPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIMinBiasZDC = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZdcCaloPlus_ZdcCaloMinus_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIMinBiasZDC = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIBptxXOR = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxXOR" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIBptxXOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1AlgoBptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG2_BptxAND OR L1_SingleEG5_BptxAND OR L1_SingleJet20U_BptxAND OR L1_SingleJet30U_BptxAND OR L1_SingleJet40U_BptxAND OR L1_SingleJet50U_BptxAND OR L1_SingleMu3_BptxAND OR L1_ZdcCaloMinus_BptxAND OR L1_ZdcCaloPlus_BptxAND OR L1_ZdcCaloPlus_ZdcCaloMinus_BptxAND OR L1_ZdcScintLooseVertex_BptxAND OR L1_ZdcScintTightVertex_BptxAND OR L1_BscMinBiasInnerThreshold1_BptxAND OR L1_BscMinBiasInnerThreshold2_BptxAND OR L1_BscMinBiasOR_BptxAND OR L1_BscMinBiasThreshold1_BptxAND OR L1_BscMinBiasThreshold2_BptxAND OR L1_DoubleEG5_BptxAND OR L1_DoubleMuOpen_BptxAND OR L1_ETT30_BptxAND OR L1_ETT60_BptxAND OR L1_HcalHfCoincidencePm_BptxAND OR L1_HcalHfMmOrPpOrPm_BptxAND OR L1_HcalHfMmpOrMpp_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1AlgoBptxAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1AlgoBptxANDNotBSC2 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_NotBsc2_BptxAND_BscMinBiasInnerThreshold1 OR L1_NotBsc2_BptxAND_BscMinBiasInnerThreshold2 OR L1_NotBsc2_BptxAND_BscMinBiasOR OR L1_NotBsc2_BptxAND_BscMinBiasThreshold1 OR L1_NotBsc2_BptxAND_BscMinBiasThreshold2 OR L1_NotBsc2_BptxAND_HcalHfCoincidencePm OR L1_NotBsc2_BptxAND_HcalHfMmOrPpOrPm OR L1_NotBsc2_BptxAND_HcalHfMmpOrMpp" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1AlgoBptxANDNotBSC2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1HFtech = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( " L1_HcalHfMmOrPpOrPm  OR L1_HcalHfCoincidencePm OR L1_HcalHfMmpOrMpp" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1HFTech = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1AlgoNotBSC2 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_NotBsc2_BscMinBiasInnerThreshold1 OR L1_NotBsc2_BscMinBiasInnerThreshold2 OR L1_NotBsc2_BscMinBiasOR OR L1_NotBsc2_BscMinBiasThreshold1 OR L1_NotBsc2_BscMinBiasThreshold2 OR L1_NotBsc2_HcalHfCoincidencePm OR L1_NotBsc2_HcalHfMmOrPpOrPm OR L1_NotBsc2_HcalHfMmpOrMpp" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1AlgoNotBSC2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1AlgoUnprotected = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HcalHfCoincidencePm OR L1_HcalHfMmOrPpOrPm OR L1_HcalHfMmpOrMpp OR L1_BscMinBiasInnerThreshold1 OR L1_BscMinBiasThreshold1 OR L1_DoubleJet30U OR L1_DoubleEG2 OR L1_DoubleEG5 OR L1_DoubleMu3 OR L1_DoubleMuOpen OR L1_SingleMu3 OR L1_ETT30" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1AlgoUnprotected = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1AlgoZDC = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZdcCaloMinus OR L1_ZdcCaloPlus OR L1_ZdcCaloPlus_ZdcCaloMinus OR L1_ZdcScintMinus OR L1_ZdcScintPlus OR L1_ZdcScintTightVertex" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1AlgoZDC = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltBPTXCoincidence = cms.EDFilter( "HLTLevel1Activity",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    daqPartitions = cms.uint32( 1 ),
    ignoreL1Mask = cms.bool( True ),
    invert = cms.bool( False ),
    bunchCrossings = cms.vint32( 0, -1, 1 ),
    physicsLoBits = cms.uint64( 0x1 ),
    physicsHiBits = cms.uint64( 0x40000 ),
    technicalBits = cms.uint64( 0x1 )
)
process.hltL1sHIL1SingleMu3 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1SingleMu3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIL1SingleMu3Core = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1SingleMu5 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu5" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1SingleMu5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIL1SingleMu5Core = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1SingleMu7 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1SingleMu7 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIL1SingleMu7Core = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1DoubleMuOpen = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMuOpen_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1DoubleMuOpen = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltHIDoubleMuLevel1PathL1OpenFiltered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    PreviousCandTag = cms.InputTag( "hltL1sHIL1DoubleMuOpen" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    ExcludeSingleSegmentCSC = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    SaveTag = cms.untracked.bool( True ),
    SelectQualities = cms.vint32(  )
)
process.hltPreHIL1DoubleMuOpenCore = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1ETT30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETT30_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1ETT30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIL1ETT30Core = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1ETT60 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETT60" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1ETT60 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIL1ETT60Core = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1TechBSCminBias = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "32 OR 33 OR 40 OR 41" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1TechBSCminBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1TechBSCminBiasOR = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "34" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1TechBSCminBiasOR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHighMultiplicityBSC = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "35" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIL1TechBSCHighMultiplicity = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIUpcEcal = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "NOT L1_ETT60 AND L1_DoubleEG5_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIUpcEcal = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIUpcEcalCore = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIUpcMu = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "NOT L1_ETT60 AND L1_SingleMu3_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIUpcMu = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIUpcMuCore = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIPhoton15 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIPhoton15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltEcalRawToRecHitFacility = cms.EDProducer( "EcalRawToRecHitFacility",
    sourceTag = cms.InputTag( "rawDataCollector" ),
    workerName = cms.string( "" )
)
process.hltEcalRegionalRestFEDs = cms.EDProducer( "EcalRawToRecHitRoI",
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
process.hltEcalRecHitAll = cms.EDProducer( "EcalRawToRecHitProducer",
    lazyGetterTag = cms.InputTag( "hltEcalRawToRecHitFacility" ),
    sourceTag = cms.InputTag( "hltEcalRegionalRestFEDs" ),
    splitOutput = cms.bool( True ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    rechitCollection = cms.string( "NotNeededsplitOutputTrue" )
)
process.hltHcalDigis = cms.EDProducer( "HcalRawToDigi",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    UnpackCalib = cms.untracked.bool( True ),
    UnpackZDC = cms.untracked.bool( True ),
    firstSample = cms.int32( 0 ),
    lastSample = cms.int32( 9 ),
    FilterDataQuality = cms.bool( True )
)
process.hltHbhereco = cms.EDProducer( "HcalHitReconstructor",
    firstSample = cms.int32( 4 ),
    samplesToAdd = cms.int32( 4 ),
    correctForTimeslew = cms.bool( True ),
    correctForPhaseContainment = cms.bool( True ),
    correctionPhaseNS = cms.double( 13.0 ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    correctTiming = cms.bool( False ),
    setNoiseFlags = cms.bool( False ),
    setHSCPFlags = cms.bool( False ),
    setSaturationFlags = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    dropZSmarkedPassed = cms.bool( True ),
    Subdetector = cms.string( "HBHE" ),
    setTimingShapedCutsFlags = cms.bool( False ),
    timingshapedcutsParameters = cms.PSet( 
      tfilterEnvelope = cms.vdouble( 4.0, 12.04, 13.0, 10.56, 23.5, 8.82, 37.0, 7.38, 56.0, 6.3, 81.0, 5.64, 114.5, 5.44, 175.5, 5.38, 350.5, 5.14 ),
      ignorelowest = cms.bool( True ),
      ignorehighest = cms.bool( False ),
      win_offset = cms.double( 0.0 ),
      win_gain = cms.double( 1.0 )
    ),
    flagParameters = cms.PSet( 
      nominalPedestal = cms.double( 3.0 ),
      hitEnergyMinimum = cms.double( 1.0 ),
      hitMultiplicityThreshold = cms.int32( 17 ),
      pulseShapeParameterSets = cms.VPSet( 
        cms.PSet(  pulseShapeParameters = cms.vdouble( 0.0, 100.0, -50.0, 0.0, -15.0, 0.15 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 100.0, 2000.0, -50.0, 0.0, -5.0, 0.05 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 2000.0, 1000000.0, -50.0, 0.0, 95.0, 0.0 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( -1000000.0, 1000000.0, 45.0, 0.1, 1000000.0, 0.0 )        )
      )
    ),
    hfTimingTrustParameters = cms.PSet( 
      hfTimingTrustLevel1 = cms.int32( 1 ),
      hfTimingTrustLevel2 = cms.int32( 4 )
    ),
    digistat = cms.PSet(  ),
    HFInWindowStat = cms.PSet(  ),
    S8S1stat = cms.PSet( 
      short_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      shortEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      longEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      flagsToSkip = cms.int32( 16 ),
      isS8S1 = cms.bool( True )
    ),
    PETstat = cms.PSet( 
      short_R = cms.vdouble( 0.8 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_R = cms.vdouble( 0.98 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      flagsToSkip = cms.int32( 0 ),
      short_R_29 = cms.vdouble( 0.8 ),
      long_R_29 = cms.vdouble( 0.8 )
    ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    hscpParameters = cms.PSet( 
      r1Min = cms.double( 0.15 ),
      r1Max = cms.double( 1.0 ),
      r2Min = cms.double( 0.1 ),
      r2Max = cms.double( 0.5 ),
      fracLeaderMin = cms.double( 0.4 ),
      fracLeaderMax = cms.double( 0.7 ),
      slopeMin = cms.double( -1.5 ),
      slopeMax = cms.double( -0.6 ),
      outerMin = cms.double( 0.0 ),
      outerMax = cms.double( 0.1 ),
      TimingEnergyThreshold = cms.double( 30.0 )
    ),
    S9S1stat = cms.PSet( 
      short_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      flagsToSkip = cms.int32( 24 ),
      isS8S1 = cms.bool( False )
    ),
    firstAuxOffset = cms.int32( 0 )
)
process.hltHfreco = cms.EDProducer( "HcalHitReconstructor",
    firstSample = cms.int32( 3 ),
    samplesToAdd = cms.int32( 4 ),
    correctForTimeslew = cms.bool( False ),
    correctForPhaseContainment = cms.bool( False ),
    correctionPhaseNS = cms.double( 0.0 ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    correctTiming = cms.bool( False ),
    setNoiseFlags = cms.bool( False ),
    setHSCPFlags = cms.bool( False ),
    setSaturationFlags = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    dropZSmarkedPassed = cms.bool( True ),
    Subdetector = cms.string( "HF" ),
    setTimingShapedCutsFlags = cms.bool( False ),
    timingshapedcutsParameters = cms.PSet( 
      tfilterEnvelope = cms.vdouble( 4.0, 12.04, 13.0, 10.56, 23.5, 8.82, 37.0, 7.38, 56.0, 6.3, 81.0, 5.64, 114.5, 5.44, 175.5, 5.38, 350.5, 5.14 ),
      ignorelowest = cms.bool( True ),
      ignorehighest = cms.bool( False ),
      win_offset = cms.double( 0.0 ),
      win_gain = cms.double( 1.0 )
    ),
    flagParameters = cms.PSet( 
      nominalPedestal = cms.double( 3.0 ),
      hitEnergyMinimum = cms.double( 1.0 ),
      hitMultiplicityThreshold = cms.int32( 17 ),
      pulseShapeParameterSets = cms.VPSet( 
        cms.PSet(  pulseShapeParameters = cms.vdouble( 0.0, 100.0, -50.0, 0.0, -15.0, 0.15 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 100.0, 2000.0, -50.0, 0.0, -5.0, 0.05 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 2000.0, 1000000.0, -50.0, 0.0, 95.0, 0.0 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( -1000000.0, 1000000.0, 45.0, 0.1, 1000000.0, 0.0 )        )
      )
    ),
    hfTimingTrustParameters = cms.PSet( 
      hfTimingTrustLevel1 = cms.int32( 1 ),
      hfTimingTrustLevel2 = cms.int32( 4 )
    ),
    digistat = cms.PSet( 
      HFdigiflagFirstSample = cms.int32( 3 ),
      HFdigiflagSamplesToAdd = cms.int32( 4 ),
      HFdigiflagExpectedPeak = cms.int32( 4 ),
      HFdigiflagMinEthreshold = cms.double( 40.0 ),
      HFdigiflagCoef0 = cms.double( 0.93 ),
      HFdigiflagCoef1 = cms.double( -0.38275 ),
      HFdigiflagCoef2 = cms.double( -0.012667 )
    ),
    HFInWindowStat = cms.PSet( 
      hflongMinWindowTime = cms.vdouble( -10.0 ),
      hflongMaxWindowTime = cms.vdouble( 10.0 ),
      hflongEthresh = cms.double( 40.0 ),
      hfshortMinWindowTime = cms.vdouble( -12.0 ),
      hfshortMaxWindowTime = cms.vdouble( 10.0 ),
      hfshortEthresh = cms.double( 40.0 )
    ),
    S8S1stat = cms.PSet( 
      short_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      shortEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      longEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      flagsToSkip = cms.int32( 16 ),
      isS8S1 = cms.bool( True )
    ),
    PETstat = cms.PSet( 
      short_R = cms.vdouble( 0.8 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_R = cms.vdouble( 0.98 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      flagsToSkip = cms.int32( 0 ),
      short_R_29 = cms.vdouble( 0.8 ),
      long_R_29 = cms.vdouble( 0.8 )
    ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    hscpParameters = cms.PSet( 
      r1Min = cms.double( 0.15 ),
      r1Max = cms.double( 1.0 ),
      r2Min = cms.double( 0.1 ),
      r2Max = cms.double( 0.5 ),
      fracLeaderMin = cms.double( 0.4 ),
      fracLeaderMax = cms.double( 0.7 ),
      slopeMin = cms.double( -1.5 ),
      slopeMax = cms.double( -0.6 ),
      outerMin = cms.double( 0.0 ),
      outerMax = cms.double( 0.1 ),
      TimingEnergyThreshold = cms.double( 30.0 )
    ),
    S9S1stat = cms.PSet( 
      short_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      flagsToSkip = cms.int32( 24 ),
      isS8S1 = cms.bool( False )
    ),
    firstAuxOffset = cms.int32( 0 )
)
process.hltHoreco = cms.EDProducer( "HcalHitReconstructor",
    firstSample = cms.int32( 4 ),
    samplesToAdd = cms.int32( 4 ),
    correctForTimeslew = cms.bool( True ),
    correctForPhaseContainment = cms.bool( True ),
    correctionPhaseNS = cms.double( 13.0 ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    correctTiming = cms.bool( False ),
    setNoiseFlags = cms.bool( False ),
    setHSCPFlags = cms.bool( False ),
    setSaturationFlags = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    dropZSmarkedPassed = cms.bool( True ),
    Subdetector = cms.string( "HO" ),
    setTimingShapedCutsFlags = cms.bool( False ),
    timingshapedcutsParameters = cms.PSet( 
      tfilterEnvelope = cms.vdouble( 4.0, 12.04, 13.0, 10.56, 23.5, 8.82, 37.0, 7.38, 56.0, 6.3, 81.0, 5.64, 114.5, 5.44, 175.5, 5.38, 350.5, 5.14 ),
      ignorelowest = cms.bool( True ),
      ignorehighest = cms.bool( False ),
      win_offset = cms.double( 0.0 ),
      win_gain = cms.double( 1.0 )
    ),
    flagParameters = cms.PSet( 
      nominalPedestal = cms.double( 3.0 ),
      hitEnergyMinimum = cms.double( 1.0 ),
      hitMultiplicityThreshold = cms.int32( 17 ),
      pulseShapeParameterSets = cms.VPSet( 
        cms.PSet(  pulseShapeParameters = cms.vdouble( 0.0, 100.0, -50.0, 0.0, -15.0, 0.15 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 100.0, 2000.0, -50.0, 0.0, -5.0, 0.05 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 2000.0, 1000000.0, -50.0, 0.0, 95.0, 0.0 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( -1000000.0, 1000000.0, 45.0, 0.1, 1000000.0, 0.0 )        )
      )
    ),
    hfTimingTrustParameters = cms.PSet( 
      hfTimingTrustLevel1 = cms.int32( 1 ),
      hfTimingTrustLevel2 = cms.int32( 4 )
    ),
    digistat = cms.PSet(  ),
    HFInWindowStat = cms.PSet(  ),
    S8S1stat = cms.PSet( 
      short_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      shortEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      longEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      flagsToSkip = cms.int32( 16 ),
      isS8S1 = cms.bool( True )
    ),
    PETstat = cms.PSet( 
      short_R = cms.vdouble( 0.8 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_R = cms.vdouble( 0.98 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      flagsToSkip = cms.int32( 0 ),
      short_R_29 = cms.vdouble( 0.8 ),
      long_R_29 = cms.vdouble( 0.8 )
    ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    hscpParameters = cms.PSet( 
      r1Min = cms.double( 0.15 ),
      r1Max = cms.double( 1.0 ),
      r2Min = cms.double( 0.1 ),
      r2Max = cms.double( 0.5 ),
      fracLeaderMin = cms.double( 0.4 ),
      fracLeaderMax = cms.double( 0.7 ),
      slopeMin = cms.double( -1.5 ),
      slopeMax = cms.double( -0.6 ),
      outerMin = cms.double( 0.0 ),
      outerMax = cms.double( 0.1 ),
      TimingEnergyThreshold = cms.double( 30.0 )
    ),
    S9S1stat = cms.PSet( 
      short_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      flagsToSkip = cms.int32( 24 ),
      isS8S1 = cms.bool( False )
    ),
    firstAuxOffset = cms.int32( 0 )
)
process.hltTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
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
    HcalAcceptSeverityLevel = cms.uint32( 11 ),
    EcalAcceptSeverityLevel = cms.uint32( 3 ),
    UseHcalRecoveredHits = cms.bool( True ),
    UseEcalRecoveredHits = cms.bool( True ),
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
    ecalInputs = cms.VInputTag( 'hltEcalRecHitAll:EcalRecHitsEB','hltEcalRecHitAll:EcalRecHitsEE' )
)
process.hltIslandBasicClustersHI = cms.EDProducer( "IslandClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    barrelHitProducer = cms.string( "hltEcalRecHitAll" ),
    endcapHitProducer = cms.string( "hltEcalRecHitAll" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClustersHI" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClustersHI" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0_barl = cms.double( 7.4 ),
    posCalc_t0_endc = cms.double( 3.1 ),
    posCalc_t0_endcPresh = cms.double( 1.2 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 ),
    clustershapecollectionEB = cms.string( "islandBarrelShape" ),
    clustershapecollectionEE = cms.string( "islandEndcapShape" ),
    barrelShapeAssociation = cms.string( "islandBarrelShapeAssoc" ),
    endcapShapeAssociation = cms.string( "islandEndcapShapeAssoc" )
)
process.hltIslandSuperClustersHI = cms.EDProducer( "SuperClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    endcapClusterProducer = cms.string( "hltIslandBasicClustersHI" ),
    barrelClusterProducer = cms.string( "hltIslandBasicClustersHI" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClustersHI" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClustersHI" ),
    endcapSuperclusterCollection = cms.string( "islandEndcapSuperClustersHI" ),
    barrelSuperclusterCollection = cms.string( "islandBarrelSuperClustersHI" ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( True ),
    barrelEtaSearchRoad = cms.double( 0.06 ),
    barrelPhiSearchRoad = cms.double( 0.8 ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    endcapPhiSearchRoad = cms.double( 0.6 ),
    seedTransverseEnergyThreshold = cms.double( 1.0 )
)
process.hltCorrectedIslandBarrelSuperClustersHI = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( 'hltIslandSuperClustersHI','islandBarrelSuperClustersHI' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( False ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 0.0 ),
      fBremVec = cms.vdouble( 0.0 ),
      brLinearHighThr = cms.double( 0.0 ),
      fEtEtaVec = cms.vdouble( 0.0 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
process.hltCorrectedIslandEndcapSuperClustersHI = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    rawSuperClusterProducer = cms.InputTag( 'hltIslandSuperClustersHI','islandEndcapSuperClustersHI' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( False ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 0.0 ),
      fBremVec = cms.vdouble( 0.0 ),
      brLinearHighThr = cms.double( 0.0 ),
      fEtEtaVec = cms.vdouble( 0.0 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
process.hltRecoHIEcalCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag( "hltCorrectedIslandBarrelSuperClustersHI" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedIslandEndcapSuperClustersHI" ),
    recoEcalCandidateCollection = cms.string( "" )
)
process.hltHIPhoton15 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalCandidate" ),
    MinPt = cms.double( 15.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 1 )
)
process.hltPreHIPhoton15Core = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIPhoton20 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIPhoton20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltHIPhoton20 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalCandidate" ),
    MinPt = cms.double( 20.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 1 )
)
process.hltPreHIPhoton20Core = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIPhoton30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIPhoton30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltHIPhoton30 = cms.EDFilter( "HLT1Photon",
    inputTag = cms.InputTag( "hltRecoHIEcalCandidate" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 2.0 ),
    MinN = cms.int32( 1 )
)
process.hltPreHIPhoton30Core = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIL1DoubleEG5 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG5_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIDoublePhoton5CEPL1R = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltESRawToRecHitFacility = cms.EDProducer( "EcalRawToRecHitFacility",
    sourceTag = cms.InputTag( "rawDataCollector" ),
    workerName = cms.string( "esRawToRecHit" )
)
process.hltEcalRegionalEgammaFEDs = cms.EDProducer( "EcalRawToRecHitRoI",
    sourceTag = cms.InputTag( "hltEcalRawToRecHitFacility" ),
    type = cms.string( "egamma" ),
    doES = cms.bool( True ),
    sourceTag_es = cms.InputTag( "hltESRawToRecHitFacility" ),
    MuJobPSet = cms.PSet(  ),
    JetJobPSet = cms.VPSet( 
    ),
    EmJobPSet = cms.VPSet( 
      cms.PSet(  Source = cms.InputTag( 'hltL1extraParticles','Isolated' ),
        regionPhiMargin = cms.double( 0.4 ),
        Ptmin = cms.double( 5.0 ),
        regionEtaMargin = cms.double( 0.25 )
      ),
      cms.PSet(  Source = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
        regionPhiMargin = cms.double( 0.4 ),
        Ptmin = cms.double( 5.0 ),
        regionEtaMargin = cms.double( 0.25 )
      )
    ),
    CandJobPSet = cms.VPSet( 
    )
)
process.hltEcalRegionalEgammaRecHit = cms.EDProducer( "EcalRawToRecHitProducer",
    lazyGetterTag = cms.InputTag( "hltEcalRawToRecHitFacility" ),
    sourceTag = cms.InputTag( "hltEcalRegionalEgammaFEDs" ),
    splitOutput = cms.bool( True ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    rechitCollection = cms.string( "NotNeededsplitOutputTrue" )
)
process.hltESRegionalEgammaRecHit = cms.EDProducer( "EcalRawToRecHitProducer",
    lazyGetterTag = cms.InputTag( "hltESRawToRecHitFacility" ),
    sourceTag = cms.InputTag( 'hltEcalRegionalEgammaFEDs','es' ),
    splitOutput = cms.bool( False ),
    EBrechitCollection = cms.string( "" ),
    EErechitCollection = cms.string( "" ),
    rechitCollection = cms.string( "EcalRecHitsES" )
)
process.hltHybridSuperClustersL1Isolated = cms.EDProducer( "EgammaHLTHybridClusterProducer",
    debugLevel = cms.string( "INFO" ),
    basicclusterCollection = cms.string( "" ),
    superclusterCollection = cms.string( "" ),
    ecalhitproducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalhitcollection = cms.string( "EcalRecHitsEB" ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    doIsolated = cms.bool( True ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.14 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0 = cms.double( 7.4 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 ),
    HybridBarrelSeedThr = cms.double( 1.5 ),
    step = cms.int32( 17 ),
    ethresh = cms.double( 0.1 ),
    eseed = cms.double( 0.35 ),
    ewing = cms.double( 0.0 ),
    dynamicEThresh = cms.bool( False ),
    eThreshA = cms.double( 0.0030 ),
    eThreshB = cms.double( 0.1 ),
    severityRecHitThreshold = cms.double( 4.0 ),
    severitySpikeId = cms.int32( 2 ),
    severitySpikeThreshold = cms.double( 0.95 ),
    excludeFlagged = cms.bool( False ),
    dynamicPhiRoad = cms.bool( False ),
    RecHitFlagToBeExcluded = cms.vint32(  ),
    RecHitSeverityToBeExcluded = cms.vint32( 999 ),
    bremRecoveryPset = cms.PSet(  )
)
process.hltCorrectedHybridSuperClustersL1Isolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( "hltHybridSuperClustersL1Isolated" ),
    superClusterAlgo = cms.string( "Hybrid" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 1.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 1.1 ),
      fEtEtaVec = cms.vdouble( 1.0012, -0.5714, 0.0, 0.0, 0.0, 0.5549, 12.74, 1.0448, 0.0, 0.0, 0.0, 0.0, 8.0, 1.023, -0.00181, 0.0, 0.0 ),
      brLinearHighThr = cms.double( 8.0 ),
      fBremVec = cms.vdouble( -0.05208, 0.1331, 0.9196, -5.735E-4, 1.343 )
    ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
process.hltMulti5x5BasicClustersL1Isolated = cms.EDProducer( "EgammaHLTMulti5x5ClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    doIsolated = cms.bool( True ),
    barrelHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    endcapHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "notused" ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    Multi5x5BarrelSeedThr = cms.double( 0.5 ),
    Multi5x5EndcapSeedThr = cms.double( 0.18 ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.3 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0_barl = cms.double( 7.4 ),
    posCalc_t0_endc = cms.double( 3.1 ),
    posCalc_t0_endcPresh = cms.double( 1.2 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 ),
    RecHitFlagToBeExcluded = cms.vint32(  )
)
process.hltMulti5x5SuperClustersL1Isolated = cms.EDProducer( "Multi5x5SuperClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    endcapClusterProducer = cms.string( "hltMulti5x5BasicClustersL1Isolated" ),
    barrelClusterProducer = cms.string( "notused" ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    barrelClusterCollection = cms.string( "multi5x5BarrelBasicClusters" ),
    endcapSuperclusterCollection = cms.string( "multi5x5EndcapSuperClusters" ),
    barrelSuperclusterCollection = cms.string( "multi5x5BarrelSuperClusters" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    barrelEtaSearchRoad = cms.double( 0.06 ),
    barrelPhiSearchRoad = cms.double( 0.8 ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    endcapPhiSearchRoad = cms.double( 0.6 ),
    seedTransverseEnergyThreshold = cms.double( 1.0 ),
    dynamicPhiRoad = cms.bool( False ),
    bremRecoveryPset = cms.PSet( 
      barrel = cms.PSet(  ),
      endcap = cms.PSet( 
        a = cms.double( 47.85 ),
        c = cms.double( 0.1201 ),
        b = cms.double( 108.8 )
      ),
      doEndcaps = cms.bool( True ),
      doBarrel = cms.bool( False )
    )
)
process.hltMulti5x5EndcapSuperClustersWithPreshowerL1Isolated = cms.EDProducer( "PreshowerClusterProducer",
    preshRecHitProducer = cms.InputTag( 'hltESRegionalEgammaRecHit','EcalRecHitsES' ),
    endcapSClusterProducer = cms.InputTag( 'hltMulti5x5SuperClustersL1Isolated','multi5x5EndcapSuperClusters' ),
    preshClusterCollectionX = cms.string( "preshowerXClusters" ),
    preshClusterCollectionY = cms.string( "preshowerYClusters" ),
    preshNclust = cms.int32( 4 ),
    etThresh = cms.double( 5.0 ),
    assocSClusterCollection = cms.string( "" ),
    preshStripEnergyCut = cms.double( 0.0 ),
    preshSeededNstrip = cms.int32( 15 ),
    preshClusterEnergyCut = cms.double( 0.0 ),
    debugLevel = cms.string( "" )
)
process.hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
    rawSuperClusterProducer = cms.InputTag( "hltMulti5x5EndcapSuperClustersWithPreshowerL1Isolated" ),
    superClusterAlgo = cms.string( "Multi5x5" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    etThresh = cms.double( 1.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 0.6 ),
      fEtEtaVec = cms.vdouble( 0.9746, -6.512, 0.0, 0.0, 0.02771, 4.983, 0.0, 0.0, -0.007288, -0.9446, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 ),
      brLinearHighThr = cms.double( 6.0 ),
      fBremVec = cms.vdouble( -0.04163, 0.08552, 0.95048, -0.002308, 1.077 )
    )
)
process.hltHybridSuperClustersL1NonIsolated = cms.EDProducer( "EgammaHLTHybridClusterProducer",
    debugLevel = cms.string( "INFO" ),
    basicclusterCollection = cms.string( "" ),
    superclusterCollection = cms.string( "" ),
    ecalhitproducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalhitcollection = cms.string( "EcalRecHitsEB" ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    doIsolated = cms.bool( False ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.14 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0 = cms.double( 7.4 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 ),
    HybridBarrelSeedThr = cms.double( 1.5 ),
    step = cms.int32( 17 ),
    ethresh = cms.double( 0.1 ),
    eseed = cms.double( 0.35 ),
    ewing = cms.double( 0.0 ),
    dynamicEThresh = cms.bool( False ),
    eThreshA = cms.double( 0.0030 ),
    eThreshB = cms.double( 0.1 ),
    severityRecHitThreshold = cms.double( 4.0 ),
    severitySpikeId = cms.int32( 2 ),
    severitySpikeThreshold = cms.double( 0.95 ),
    excludeFlagged = cms.bool( False ),
    dynamicPhiRoad = cms.bool( False ),
    RecHitFlagToBeExcluded = cms.vint32(  ),
    RecHitSeverityToBeExcluded = cms.vint32( 999 ),
    bremRecoveryPset = cms.PSet(  )
)
process.hltCorrectedHybridSuperClustersL1NonIsolatedTemp = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( "hltHybridSuperClustersL1NonIsolated" ),
    superClusterAlgo = cms.string( "Hybrid" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 1.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 1.1 ),
      fEtEtaVec = cms.vdouble( 1.0012, -0.5714, 0.0, 0.0, 0.0, 0.5549, 12.74, 1.0448, 0.0, 0.0, 0.0, 0.0, 8.0, 1.023, -0.00181, 0.0, 0.0 ),
      brLinearHighThr = cms.double( 8.0 ),
      fBremVec = cms.vdouble( -0.05208, 0.1331, 0.9196, -5.735E-4, 1.343 )
    ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
process.hltCorrectedHybridSuperClustersL1NonIsolated = cms.EDProducer( "EgammaHLTRemoveDuplicatedSC",
    L1NonIsoUskimmedSC = cms.InputTag( "hltCorrectedHybridSuperClustersL1NonIsolatedTemp" ),
    L1IsoSC = cms.InputTag( "hltCorrectedHybridSuperClustersL1Isolated" ),
    L1NonIsoSkimmedCollection = cms.string( "" )
)
process.hltMulti5x5BasicClustersL1NonIsolated = cms.EDProducer( "EgammaHLTMulti5x5ClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    doIsolated = cms.bool( False ),
    barrelHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    endcapHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "notused" ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    Multi5x5BarrelSeedThr = cms.double( 0.5 ),
    Multi5x5EndcapSeedThr = cms.double( 0.18 ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.3 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0_barl = cms.double( 7.4 ),
    posCalc_t0_endc = cms.double( 3.1 ),
    posCalc_t0_endcPresh = cms.double( 1.2 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 ),
    RecHitFlagToBeExcluded = cms.vint32(  )
)
process.hltMulti5x5SuperClustersL1NonIsolated = cms.EDProducer( "Multi5x5SuperClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    endcapClusterProducer = cms.string( "hltMulti5x5BasicClustersL1NonIsolated" ),
    barrelClusterProducer = cms.string( "notused" ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    barrelClusterCollection = cms.string( "multi5x5BarrelBasicClusters" ),
    endcapSuperclusterCollection = cms.string( "multi5x5EndcapSuperClusters" ),
    barrelSuperclusterCollection = cms.string( "multi5x5BarrelSuperClusters" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    barrelEtaSearchRoad = cms.double( 0.06 ),
    barrelPhiSearchRoad = cms.double( 0.8 ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    endcapPhiSearchRoad = cms.double( 0.6 ),
    seedTransverseEnergyThreshold = cms.double( 1.0 ),
    dynamicPhiRoad = cms.bool( False ),
    bremRecoveryPset = cms.PSet( 
      barrel = cms.PSet(  ),
      endcap = cms.PSet( 
        a = cms.double( 47.85 ),
        c = cms.double( 0.1201 ),
        b = cms.double( 108.8 )
      ),
      doEndcaps = cms.bool( True ),
      doBarrel = cms.bool( False )
    )
)
process.hltMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated = cms.EDProducer( "PreshowerClusterProducer",
    preshRecHitProducer = cms.InputTag( 'hltESRegionalEgammaRecHit','EcalRecHitsES' ),
    endcapSClusterProducer = cms.InputTag( 'hltMulti5x5SuperClustersL1NonIsolated','multi5x5EndcapSuperClusters' ),
    preshClusterCollectionX = cms.string( "preshowerXClusters" ),
    preshClusterCollectionY = cms.string( "preshowerYClusters" ),
    preshNclust = cms.int32( 4 ),
    etThresh = cms.double( 5.0 ),
    assocSClusterCollection = cms.string( "" ),
    preshStripEnergyCut = cms.double( 0.0 ),
    preshSeededNstrip = cms.int32( 15 ),
    preshClusterEnergyCut = cms.double( 0.0 ),
    debugLevel = cms.string( "" )
)
process.hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolatedTemp = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
    rawSuperClusterProducer = cms.InputTag( "hltMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated" ),
    superClusterAlgo = cms.string( "Multi5x5" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    etThresh = cms.double( 1.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 0.6 ),
      fEtEtaVec = cms.vdouble( 0.9746, -6.512, 0.0, 0.0, 0.02771, 4.983, 0.0, 0.0, -0.007288, -0.9446, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 ),
      brLinearHighThr = cms.double( 6.0 ),
      fBremVec = cms.vdouble( -0.04163, 0.08552, 0.95048, -0.002308, 1.077 )
    )
)
process.hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated = cms.EDProducer( "EgammaHLTRemoveDuplicatedSC",
    L1NonIsoUskimmedSC = cms.InputTag( "hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolatedTemp" ),
    L1IsoSC = cms.InputTag( "hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated" ),
    L1NonIsoSkimmedCollection = cms.string( "" )
)
process.hltL1IsoRecoEcalCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag( "hltCorrectedHybridSuperClustersL1Isolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated" ),
    recoEcalCandidateCollection = cms.string( "" )
)
process.hltL1NonIsoRecoEcalCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag( "hltCorrectedHybridSuperClustersL1NonIsolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated" ),
    recoEcalCandidateCollection = cms.string( "" )
)
process.hltHIDoublePhotonEt5L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1sHIL1DoubleEG5" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
process.hltHIDoublePhotonEt5EtPhiFilter = cms.EDFilter( "HLTEgammaDoubleEtDeltaPhiFilter",
    inputTag = cms.InputTag( "hltHIDoublePhotonEt5L1MatchFilterRegional" ),
    etcut = cms.double( 5.0 ),
    minDeltaPhi = cms.double( 2.5 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
process.hltL1IsolatedPhotonEcalIsol = cms.EDProducer( "EgammaHLTEcalRecIsolationProducer",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    ecalBarrelRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalBarrelRecHitCollection = cms.InputTag( "EcalRecHitsEB" ),
    ecalEndcapRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalEndcapRecHitCollection = cms.InputTag( "EcalRecHitsEE" ),
    etMinBarrel = cms.double( -9999.0 ),
    eMinBarrel = cms.double( 0.08 ),
    etMinEndcap = cms.double( 0.1 ),
    eMinEndcap = cms.double( -9999.0 ),
    intRadiusBarrel = cms.double( 3.0 ),
    intRadiusEndcap = cms.double( 3.0 ),
    extRadius = cms.double( 0.3 ),
    jurassicWidth = cms.double( 3.0 ),
    useIsolEt = cms.bool( True ),
    tryBoth = cms.bool( True ),
    subtract = cms.bool( False ),
    useNumCrystals = cms.bool( True )
)
process.hltL1NonIsolatedPhotonEcalIsol = cms.EDProducer( "EgammaHLTEcalRecIsolationProducer",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    ecalBarrelRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalBarrelRecHitCollection = cms.InputTag( "EcalRecHitsEB" ),
    ecalEndcapRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalEndcapRecHitCollection = cms.InputTag( "EcalRecHitsEE" ),
    etMinBarrel = cms.double( -9999.0 ),
    eMinBarrel = cms.double( 0.08 ),
    etMinEndcap = cms.double( 0.1 ),
    eMinEndcap = cms.double( -9999.0 ),
    intRadiusBarrel = cms.double( 3.0 ),
    intRadiusEndcap = cms.double( 3.0 ),
    extRadius = cms.double( 0.3 ),
    jurassicWidth = cms.double( 3.0 ),
    useIsolEt = cms.bool( True ),
    tryBoth = cms.bool( True ),
    subtract = cms.bool( False ),
    useNumCrystals = cms.bool( True )
)
process.hltHIDoublePhotonEt5EcalIsolFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltHIDoublePhotonEt5EtPhiFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    thrRegularEB = cms.double( 3.0 ),
    thrRegularEE = cms.double( 3.0 ),
    thrOverEEB = cms.double( 0.1 ),
    thrOverEEE = cms.double( 0.1 ),
    thrOverE2EB = cms.double( -1.0 ),
    thrOverE2EE = cms.double( -1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
process.hltL1IsolatedPhotonHcalForHE = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    hbheRecHitProducer = cms.InputTag( "hltHbhereco" ),
    eMinHB = cms.double( 0.7 ),
    eMinHE = cms.double( 0.8 ),
    etMinHB = cms.double( -1.0 ),
    etMinHE = cms.double( -1.0 ),
    innerCone = cms.double( 0.0 ),
    outerCone = cms.double( 0.14 ),
    depth = cms.int32( -1 ),
    doEtSum = cms.bool( False )
)
process.hltL1NonIsolatedPhotonHcalForHE = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    hbheRecHitProducer = cms.InputTag( "hltHbhereco" ),
    eMinHB = cms.double( 0.7 ),
    eMinHE = cms.double( 0.8 ),
    etMinHB = cms.double( -1.0 ),
    etMinHE = cms.double( -1.0 ),
    innerCone = cms.double( 0.0 ),
    outerCone = cms.double( 0.14 ),
    depth = cms.int32( -1 ),
    doEtSum = cms.bool( False )
)
process.hltHIDoublePhotonEt5HEFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltHIDoublePhotonEt5EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalForHE" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalForHE" ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    thrRegularEB = cms.double( 999999.9 ),
    thrRegularEE = cms.double( 999999.9 ),
    thrOverEEB = cms.double( -1.0 ),
    thrOverEEE = cms.double( -1.0 ),
    thrOverE2EB = cms.double( -1.0 ),
    thrOverE2EE = cms.double( -1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
process.hltTowerMakerForHcal = cms.EDProducer( "CaloTowersCreator",
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
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    AllowMissingInputs = cms.bool( True ),
    HcalAcceptSeverityLevel = cms.uint32( 11 ),
    EcalAcceptSeverityLevel = cms.uint32( 3 ),
    UseHcalRecoveredHits = cms.bool( True ),
    UseEcalRecoveredHits = cms.bool( True ),
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
    ecalInputs = cms.VInputTag(  )
)
process.hltHcalTowerFilter = cms.EDFilter( "HLTHcalTowerFilter",
    inputTag = cms.InputTag( "hltTowerMakerForHcal" ),
    MinE = cms.double( 5.0 ),
    MaxN = cms.int32( 10 )
)
process.hltL1sHIJet35U = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30U_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIJet35U = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltIterativeCone5PileupSubtractionCaloJets = cms.EDProducer( "FastjetJetProducer",
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
    subtractorName = cms.string( "JetOffsetCorrector" )
)
process.hltHI1jet35U = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltIterativeCone5PileupSubtractionCaloJets" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
process.hltPreHIJet35UCore = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIJet50U = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30U_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIJet50U = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltHI1jet50U = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltIterativeCone5PileupSubtractionCaloJets" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 50.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
process.hltPreHIJet50UCore = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIJet75U = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30U_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIJet75U = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltHI1jet75U = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltIterativeCone5PileupSubtractionCaloJets" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 75.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
process.hltPreHIJet75UCore = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltL1sHIJet90U = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30U_BptxAND" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIJet90U = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltHI1jet90U = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltIterativeCone5PileupSubtractionCaloJets" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 90.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
process.hltPreHIJet90UCopy = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltBPTXAntiCoincidence = cms.EDFilter( "HLTLevel1Activity",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    daqPartitions = cms.uint32( 1 ),
    ignoreL1Mask = cms.bool( True ),
    invert = cms.bool( True ),
    bunchCrossings = cms.vint32( 0, 1, -1 ),
    physicsLoBits = cms.uint64( 0x1 ),
    physicsHiBits = cms.uint64( 0x40000 ),
    technicalBits = cms.uint64( 0x1 )
)
process.hltL1sHIL1SingleJet20 = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet20U" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    saveTags = cms.untracked.bool( False )
)
process.hltPreHIStoppedHSCP35 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltStoppedHSCPHpdFilter = cms.EDFilter( "HLTHPDFilter",
    inputTag = cms.InputTag( "hltHbhereco" ),
    energy = cms.double( -99.0 ),
    hpdSpikeEnergy = cms.double( 10.0 ),
    hpdSpikeIsolationEnergy = cms.double( 1.0 ),
    rbxSpikeEnergy = cms.double( 50.0 ),
    rbxSpikeUnbalance = cms.double( 0.2 )
)
process.hltStoppedHSCPTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
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
    HcalAcceptSeverityLevel = cms.uint32( 11 ),
    EcalAcceptSeverityLevel = cms.uint32( 3 ),
    UseHcalRecoveredHits = cms.bool( True ),
    UseEcalRecoveredHits = cms.bool( True ),
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
    ecalInputs = cms.VInputTag(  )
)
process.hltStoppedHSCPIterativeCone5CaloJets = cms.EDProducer( "FastjetJetProducer",
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
    subtractorName = cms.string( "" )
)
process.hltStoppedHSCP1CaloJetEnergy35 = cms.EDFilter( "HLT1CaloJetEnergy",
    inputTag = cms.InputTag( "hltStoppedHSCPIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    MinE = cms.double( 35.0 ),
    MaxEta = cms.double( 3.0 ),
    MinN = cms.int32( 1 )
)
process.hltL1sHIGlobalDecision = cms.EDFilter( "HLTLevel1GTSeed",
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1GlobalDecision" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltPreHIActivityL1A = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIActivityPixels = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltHIPixelActivityFilter = cms.EDFilter( "HLTPixelActivityFilter",
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 3 ),
    maxClusters = cms.uint32( 0 )
)
process.hltPreHIActivityPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIActivityHFSingle3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltHcalSimpleRecHitFilter = cms.EDFilter( "HLTHcalSimpleRecHitFilter",
    threshold = cms.double( 3.0 ),
    minNHitsNeg = cms.int32( 1 ),
    minNHitsPos = cms.int32( 1 ),
    doCoincidence = cms.bool( False ),
    HFRecHitCollection = cms.InputTag( "hltHfreco" ),
    maskedChannels = cms.vint32(  )
)
process.hltPreHIActivityHFCoincidence3 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltHcalSimpleRecHitFilterCoincidence = cms.EDFilter( "HLTHcalSimpleRecHitFilter",
    threshold = cms.double( 3.0 ),
    minNHitsNeg = cms.int32( 1 ),
    minNHitsPos = cms.int32( 1 ),
    doCoincidence = cms.bool( True ),
    HFRecHitCollection = cms.InputTag( "hltHfreco" ),
    maskedChannels = cms.vint32(  )
)
process.hltPreHIClusterVertexCompatibility = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltHIPixelClusterShapeFilter = cms.EDFilter( "HLTPixelClusterShapeFilter",
    inputTag = cms.InputTag( "hltHISiPixelRecHits" ),
    minZ = cms.double( -20.0 ),
    maxZ = cms.double( 20.05 ),
    zStep = cms.double( 0.2 ),
    nhitsTrunc = cms.int32( 150 ),
    clusterTrunc = cms.double( 2.0 ),
    clusterPars = cms.vdouble(  )
)
process.hltPreHICentralityVeto = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPixelActivityFilterCentralityVeto = cms.EDFilter( "HLTPixelActivityFilter",
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 3 ),
    maxClusters = cms.uint32( 1000 )
)
process.hltRandomEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 3 )
)
process.hltPreHIRandom = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHIHcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltHcalCalibTypeFilter = cms.EDFilter( "HLTHcalCalibTypeFilter",
    InputTag = cms.InputTag( "rawDataCollector" ),
    CalibTypes = cms.vint32( 1, 2, 3, 4, 5, 6 )
)
process.hltHcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731 )
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
process.hltBoolTrue = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
process.hltL1GtTrigReport = cms.EDAnalyzer( "L1GtTrigReport",
    UseL1GlobalTriggerRecord = cms.bool( False ),
    L1GtRecordInputTag = cms.InputTag( "hltGtDigis" )
)
process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)
process.hltPreExpress = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreExpressSmart = cms.EDFilter( "TriggerResultsFilter",
    triggerConditions = cms.vstring( 'HLT_DoubleMu5_v1',
      'HLT_Jet180U_v3',
      'HLT_L1Tech_BSC_minBias_OR / 3',
      'HLT_MET80_CenJet50U_v3',
      'HLT_Mu25_v1',
      'HLT_Photon110_NoHE_Cleaned_L1R_v1',
      'HLT_Photon70_Cleaned_L1R_v1',
      'HLT_TrackerCosmics',
      'HLT_ZeroBias / 13' ),
    hltResults = cms.InputTag( "TriggerResults" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMask = cms.bool( False ),
    daqPartitions = cms.uint32( 1 ),
    throw = cms.bool( True ),
    l1techIgnorePrescales = cms.bool( False )
)
process.hltDQML1Scalers = cms.EDAnalyzer( "L1Scalers",
    l1GtData = cms.InputTag( "hltGtDigis" ),
    fedRawData = cms.InputTag( "rawDataCollector" ),
    HFRecHitCollection = cms.InputTag( "hltHfreco" ),
    maskedChannels = cms.untracked.vint32( 8137, 8141, 8146, 8149, 8150, 8153 )
)
process.hltDQML1SeedLogicScalers = cms.EDAnalyzer( "HLTSeedL1LogicScalers",
    l1BeforeMask = cms.bool( False ),
    processname = cms.string( "HLT" ),
    L1GtDaqReadoutRecordInputTag = cms.InputTag( "hltGtDigis" ),
    L1GtRecordInputTag = cms.InputTag( "unused" ),
    DQMFolder = cms.untracked.string( "HLT/HLTSeedL1LogicScalers_EvF" ),
    monitorPaths = cms.untracked.vstring( 'HLT_L1MuOpen',
      'HLT_L1Mu',
      'HLT_Mu3',
      'HLT_L1SingleForJet',
      'HLT_SingleLooseIsoTau20',
      'HLT_MinBiasEcal' )
)
process.hltDQMHLTScalers = cms.EDAnalyzer( "HLTScalers",
    triggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)
process.hltPreDQM = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreDQMSmart = cms.EDFilter( "TriggerResultsFilter",
    triggerConditions = cms.vstring( 'HLT_Activity_CSC / 10',
      'HLT_Activity_DT / 10',
      'HLT_Activity_DT_Tuned / 10',
      'HLT_Activity_Ecal_SC17 / 10',
      'HLT_Activity_Ecal_SC7 / 10',
      'HLT_BTagMu_DiJet10U_v3',
      'HLT_BTagMu_DiJet20U_Mu5_v3',
      'HLT_BTagMu_DiJet20U_v3',
      'HLT_BTagMu_DiJet30U_Mu5_v3',
      'HLT_BTagMu_DiJet30U_v3',
      'HLT_Calibration / 10',
      'HLT_DTErrors',
      'HLT_DiJet20U_Meff180U_v3',
      'HLT_DiJet50U_PT50U_v3',
      'HLT_DiJetAve100U_v3',
      'HLT_DiJetAve140U_v3',
      'HLT_DiJetAve15U_v3',
      'HLT_DiJetAve30U_v3',
      'HLT_DiJetAve50U_v3',
      'HLT_DiJetAve70U_v3',
      'HLT_DoubleEle17_SW_L1R_v1',
      'HLT_DoubleEle4_SW_eeRes_L1R_v2',
      'HLT_DoubleEle5_SW_Upsilon_L1R_v2',
      'HLT_DoubleEle8_SW_HT70U_L1R_v1',
      'HLT_DoubleIsoTau15_OneLeg_Trk5_v4',
      'HLT_DoubleIsoTau15_Trk5_v4',
      'HLT_DoubleJet15U_ForwardBackward_v3',
      'HLT_DoubleJet25U_ForwardBackward_v3',
      'HLT_DoubleJet35U_ForwardBackward_v3',
      'HLT_DoubleMu0',
      'HLT_DoubleMu0_Quarkonium_v1',
      'HLT_DoubleMu3_HT50U_v3',
      'HLT_DoubleMu3_v2',
      'HLT_DoubleMu5_v1',
      'HLT_DoublePhoton17_SingleIsol_L1R_v1',
      'HLT_DoublePhoton22_L1R_v1',
      'HLT_DoublePhoton5_CEP_L1R_v3',
      'HLT_EcalCalibration / 0',
      'HLT_EcalOnly_SumEt160_v3',
      'HLT_Ele10_SW_EleId_HT70U_L1R_v2',
      'HLT_Ele10_SW_HT100U_L1R_v2',
      'HLT_Ele10_SW_HT70U_L1R_v2',
      'HLT_Ele10_SW_L1R_v2',
      'HLT_Ele12_SW_TighterEleId_L1R_v2',
      'HLT_Ele17_SW_Isol_L1R_v2',
      'HLT_Ele17_SW_L1R_v2',
      'HLT_Ele17_SW_TightCaloEleId_Ele8HE_L1R_v2',
      'HLT_Ele17_SW_TighterEleIdIsol_L1R_v3',
      'HLT_Ele22_SW_L1R_v2',
      'HLT_Ele22_SW_TighterCaloIdIsol_L1R_v2',
      'HLT_Ele22_SW_TighterEleId_L1R_v3',
      'HLT_Ele32_SW_TighterEleId_L1R_v2',
      'HLT_ExclDiJet30U_HFAND_v3',
      'HLT_ExclDiJet30U_HFOR_v3',
      'HLT_GlobalRunHPDNoise',
      'HLT_HT100U_v3',
      'HLT_HT130U_v3',
      'HLT_HT140U_J30U_Eta3_v3',
      'HLT_HT150U_Eta3_v3',
      'HLT_HT150U_v3',
      'HLT_HT160U_Eta3_v3',
      'HLT_HT160U_v3',
      'HLT_HT200U_v3',
      'HLT_HT50U_v3',
      'HLT_HcalCalibration / 0',
      'HLT_HcalNZS',
      'HLT_HcalPhiSym',
      'HLT_IsoEle12_PFTau15_v3',
      'HLT_IsoMu11_v4',
      'HLT_IsoMu13_v4',
      'HLT_IsoMu15_v4',
      'HLT_IsoMu17_v4',
      'HLT_IsoMu9_PFTau15_v2',
      'HLT_IsoMu9_v4',
      'HLT_IsoTrackHB_v2',
      'HLT_IsoTrackHE_v3',
      'HLT_Jet100U_v3',
      'HLT_Jet140U_v3',
      'HLT_Jet15U_HcalNoiseFiltered_v3',
      'HLT_Jet15U_v3',
      'HLT_Jet180U_v3',
      'HLT_Jet30U_v3',
      'HLT_Jet50U_v3',
      'HLT_Jet70U_v3',
      'HLT_L1DoubleMuOpen',
      'HLT_L1ETT100',
      'HLT_L1Jet10U',
      'HLT_L1Jet6U',
      'HLT_L1MET20',
      'HLT_L1Mu20',
      'HLT_L1Mu7_v1',
      'HLT_L1MuOpen_AntiBPTX_v2',
      'HLT_L1MuOpen_DT_v2',
      'HLT_L1MuOpen_v2',
      'HLT_L1SingleEG8',
      'HLT_L1Tech_BSC_HighMultiplicity',
      'HLT_L1Tech_BSC_halo',
      'HLT_L1Tech_BSC_halo_forPhysicsBackground',
      'HLT_L1Tech_BSC_minBias',
      'HLT_L1Tech_BSC_minBias_OR',
      'HLT_L1Tech_HCAL_HF',
      'HLT_L1Tech_RPC_TTU_RBst1_collisions',
      'HLT_L1_BPTX',
      'HLT_L1_BPTX_MinusOnly',
      'HLT_L1_BPTX_PlusOnly',
      'HLT_L1_BptxXOR_BscMinBiasOR',
      'HLT_L2DoubleMu0',
      'HLT_L2DoubleMu20_NoVertex_v1',
      'HLT_L2Mu0_NoVertex',
      'HLT_L2Mu30_v1',
      'HLT_L2Mu7_v1',
      'HLT_LogMonitor',
      'HLT_MET100_v3',
      'HLT_MET120_v3',
      'HLT_MET45_DiJet30U_v3',
      'HLT_MET45_v3',
      'HLT_MET65_CenJet50U_v3',
      'HLT_MET80_CenJet50U_v3',
      'HLT_MinBiasPixel_SingleTrack',
      'HLT_Mu0_TkMu0_OST_Jpsi_Tight_v3',
      'HLT_Mu0_v2',
      'HLT_Mu11',
      'HLT_Mu11_Ele8_v1',
      'HLT_Mu11_PFTau15_v2',
      'HLT_Mu13_v1',
      'HLT_Mu15_v1',
      'HLT_Mu17_v1',
      'HLT_Mu19_v1',
      'HLT_Mu21_v1',
      'HLT_Mu25_v1',
      'HLT_Mu30_NoVertex_v1',
      'HLT_Mu3_Ele8_HT70U_v1',
      'HLT_Mu3_TkMu0_OST_Jpsi_Tight_v3',
      'HLT_Mu3_Track3_Jpsi_v3',
      'HLT_Mu3_Track5_Jpsi_v3',
      'HLT_Mu3_v2',
      'HLT_Mu5',
      'HLT_Mu5_Ele13_v2',
      'HLT_Mu5_Ele17_v2',
      'HLT_Mu5_HT100U_v3',
      'HLT_Mu5_HT70U_v3',
      'HLT_Mu5_Jet50U_v3',
      'HLT_Mu5_Jet70U_v3',
      'HLT_Mu5_L2Mu0',
      'HLT_Mu5_MET45_v3',
      'HLT_Mu5_TkMu0_OST_Jpsi_Tight_v2',
      'HLT_Mu5_Track0_Jpsi_v2',
      'HLT_Mu7',
      'HLT_Mu7_Photon13_Cleaned_L1R_v1',
      'HLT_Mu8_Ele8_v1',
      'HLT_Mu9',
      'HLT_MultiVertex6_v2',
      'HLT_MultiVertex8_L1ETT60_v2',
      'HLT_Photon10_Cleaned_L1R',
      'HLT_Photon110_NoHE_Cleaned_L1R_v1',
      'HLT_Photon17_Isol_SC17HE_L1R_v1',
      'HLT_Photon20_Cleaned_L1R',
      'HLT_Photon20_Isol_Cleaned_L1R_v1',
      'HLT_Photon20_NoHE_L1R',
      'HLT_Photon22_SC22HE_L1R_v1',
      'HLT_Photon30_Cleaned_L1R',
      'HLT_Photon40_CaloId_Cleaned_L1R_v1',
      'HLT_Photon40_Isol_Cleaned_L1R_v1',
      'HLT_Photon50_Cleaned_L1R_v1',
      'HLT_Photon50_NoHE_L1R',
      'HLT_Photon70_Cleaned_L1R_v1',
      'HLT_PixelTracks_Multiplicity100',
      'HLT_PixelTracks_Multiplicity70',
      'HLT_PixelTracks_Multiplicity85',
      'HLT_QuadJet15U_v3',
      'HLT_QuadJet20U_v3',
      'HLT_QuadJet25U_v3',
      'HLT_R010U_MR50U',
      'HLT_R030U_MR100U',
      'HLT_R033U_MR100U',
      'HLT_RP025U_MR70U',
      'HLT_RPCBarrelCosmics',
      'HLT_Random',
      'HLT_SingleIsoTau20_Trk15_MET25_v4',
      'HLT_SingleIsoTau35_Trk15_MET25_v4',
      'HLT_StoppedHSCP20_v3',
      'HLT_StoppedHSCP35_v3',
      'HLT_TechTrigHCALNoise',
      'HLT_TrackerCosmics',
      'HLT_ZeroBias',
      'HLT_ZeroBiasPixel_SingleTrack' ),
    hltResults = cms.InputTag( "TriggerResults" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMask = cms.bool( False ),
    daqPartitions = cms.uint32( 1 ),
    throw = cms.bool( True ),
    l1techIgnorePrescales = cms.bool( False )
)
process.hltPreHLTDQM = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHLTDQMSmart = cms.EDFilter( "TriggerResultsFilter",
    triggerConditions = cms.vstring( 'AlCa_EcalEta / 100',
      'AlCa_EcalPhiSym / 100',
      'AlCa_EcalPi0 / 100',
      'AlCa_RPCMuonNoHits / 100',
      'AlCa_RPCMuonNoTriggers / 100',
      'AlCa_RPCMuonNormalisation / 100',
      'HLT_Activity_CSC / 10',
      'HLT_Activity_DT / 10',
      'HLT_Activity_DT_Tuned / 10',
      'HLT_Activity_Ecal_SC17 / 10',
      'HLT_Activity_Ecal_SC7 / 10',
      'HLT_BTagMu_DiJet10U_v3',
      'HLT_BTagMu_DiJet20U_Mu5_v3',
      'HLT_BTagMu_DiJet20U_v3',
      'HLT_BTagMu_DiJet30U_Mu5_v3',
      'HLT_BTagMu_DiJet30U_v3',
      'HLT_Calibration / 0',
      'HLT_DTErrors',
      'HLT_DiJet20U_Meff180U_v3',
      'HLT_DiJet50U_PT50U_v3',
      'HLT_DiJetAve100U_v3',
      'HLT_DiJetAve140U_v3',
      'HLT_DiJetAve15U_v3',
      'HLT_DiJetAve30U_v3',
      'HLT_DiJetAve50U_v3',
      'HLT_DiJetAve70U_v3',
      'HLT_DoubleEle17_SW_L1R_v1',
      'HLT_DoubleEle4_SW_eeRes_L1R_v2',
      'HLT_DoubleEle5_SW_Upsilon_L1R_v2',
      'HLT_DoubleEle8_SW_HT70U_L1R_v1',
      'HLT_DoubleIsoTau15_OneLeg_Trk5_v4',
      'HLT_DoubleIsoTau15_Trk5_v4',
      'HLT_DoubleJet15U_ForwardBackward_v3',
      'HLT_DoubleJet25U_ForwardBackward_v3',
      'HLT_DoubleJet35U_ForwardBackward_v3',
      'HLT_DoubleMu0',
      'HLT_DoubleMu0_Quarkonium_v1',
      'HLT_DoubleMu3_HT50U_v3',
      'HLT_DoubleMu3_v2',
      'HLT_DoubleMu5_v1',
      'HLT_DoublePhoton17_SingleIsol_L1R_v1',
      'HLT_DoublePhoton22_L1R_v1',
      'HLT_DoublePhoton5_CEP_L1R_v3',
      'HLT_EcalCalibration / 0',
      'HLT_EcalOnly_SumEt160_v3',
      'HLT_Ele10_SW_EleId_HT70U_L1R_v2',
      'HLT_Ele10_SW_HT100U_L1R_v2',
      'HLT_Ele10_SW_HT70U_L1R_v2',
      'HLT_Ele10_SW_L1R_v2',
      'HLT_Ele12_SW_TighterEleId_L1R_v2',
      'HLT_Ele17_SW_Isol_L1R_v2',
      'HLT_Ele17_SW_L1R_v2',
      'HLT_Ele17_SW_TightCaloEleId_Ele8HE_L1R_v2',
      'HLT_Ele17_SW_TighterEleIdIsol_L1R_v3',
      'HLT_Ele22_SW_L1R_v2',
      'HLT_Ele22_SW_TighterCaloIdIsol_L1R_v2',
      'HLT_Ele22_SW_TighterEleId_L1R_v3',
      'HLT_Ele32_SW_TighterEleId_L1R_v2',
      'HLT_ExclDiJet30U_HFAND_v3',
      'HLT_ExclDiJet30U_HFOR_v3',
      'HLT_GlobalRunHPDNoise',
      'HLT_HT100U_v3',
      'HLT_HT130U_v3',
      'HLT_HT140U_J30U_Eta3_v3',
      'HLT_HT150U_Eta3_v3',
      'HLT_HT150U_v3',
      'HLT_HT160U_Eta3_v3',
      'HLT_HT160U_v3',
      'HLT_HT200U_v3',
      'HLT_HT50U_v3',
      'HLT_HcalCalibration / 0',
      'HLT_HcalNZS',
      'HLT_HcalPhiSym',
      'HLT_IsoEle12_PFTau15_v3',
      'HLT_IsoMu11_v4',
      'HLT_IsoMu13_v4',
      'HLT_IsoMu15_v4',
      'HLT_IsoMu17_v4',
      'HLT_IsoMu9_PFTau15_v2',
      'HLT_IsoMu9_v4',
      'HLT_IsoTrackHB_v2',
      'HLT_IsoTrackHE_v3',
      'HLT_Jet100U_v3',
      'HLT_Jet140U_v3',
      'HLT_Jet15U_HcalNoiseFiltered_v3',
      'HLT_Jet15U_v3',
      'HLT_Jet180U_v3',
      'HLT_Jet30U_v3',
      'HLT_Jet50U_v3',
      'HLT_Jet70U_v3',
      'HLT_L1DoubleMuOpen',
      'HLT_L1ETT100',
      'HLT_L1Jet10U',
      'HLT_L1Jet6U',
      'HLT_L1MET20',
      'HLT_L1Mu20',
      'HLT_L1Mu7_v1',
      'HLT_L1MuOpen_AntiBPTX_v2',
      'HLT_L1MuOpen_DT_v2',
      'HLT_L1MuOpen_v2',
      'HLT_L1SingleEG8',
      'HLT_L1Tech_BSC_HighMultiplicity',
      'HLT_L1Tech_BSC_halo',
      'HLT_L1Tech_BSC_halo_forPhysicsBackground',
      'HLT_L1Tech_BSC_minBias',
      'HLT_L1Tech_BSC_minBias_OR',
      'HLT_L1Tech_HCAL_HF',
      'HLT_L1Tech_RPC_TTU_RBst1_collisions',
      'HLT_L1_BPTX',
      'HLT_L1_BPTX_MinusOnly',
      'HLT_L1_BPTX_PlusOnly',
      'HLT_L1_BptxXOR_BscMinBiasOR',
      'HLT_L2DoubleMu0',
      'HLT_L2DoubleMu20_NoVertex_v1',
      'HLT_L2Mu0_NoVertex',
      'HLT_L2Mu30_v1',
      'HLT_L2Mu7_v1',
      'HLT_LogMonitor',
      'HLT_MET100_v3',
      'HLT_MET120_v3',
      'HLT_MET45_DiJet30U_v3',
      'HLT_MET45_v3',
      'HLT_MET65_CenJet50U_v3',
      'HLT_MET80_CenJet50U_v3',
      'HLT_MinBiasPixel_SingleTrack',
      'HLT_Mu0_TkMu0_OST_Jpsi_Tight_v3',
      'HLT_Mu0_v2',
      'HLT_Mu11',
      'HLT_Mu11_Ele8_v1',
      'HLT_Mu11_PFTau15_v2',
      'HLT_Mu13_v1',
      'HLT_Mu15_v1',
      'HLT_Mu17_v1',
      'HLT_Mu19_v1',
      'HLT_Mu21_v1',
      'HLT_Mu25_v1',
      'HLT_Mu30_NoVertex_v1',
      'HLT_Mu3_Ele8_HT70U_v1',
      'HLT_Mu3_TkMu0_OST_Jpsi_Tight_v3',
      'HLT_Mu3_Track3_Jpsi_v3',
      'HLT_Mu3_Track5_Jpsi_v3',
      'HLT_Mu3_v2',
      'HLT_Mu5',
      'HLT_Mu5_Ele13_v2',
      'HLT_Mu5_Ele17_v2',
      'HLT_Mu5_HT100U_v3',
      'HLT_Mu5_HT70U_v3',
      'HLT_Mu5_Jet50U_v3',
      'HLT_Mu5_Jet70U_v3',
      'HLT_Mu5_L2Mu0',
      'HLT_Mu5_MET45_v3',
      'HLT_Mu5_TkMu0_OST_Jpsi_Tight_v2',
      'HLT_Mu5_Track0_Jpsi_v2',
      'HLT_Mu7',
      'HLT_Mu7_Photon13_Cleaned_L1R_v1',
      'HLT_Mu8_Ele8_v1',
      'HLT_Mu9',
      'HLT_MultiVertex6_v2',
      'HLT_MultiVertex8_L1ETT60_v2',
      'HLT_Photon10_Cleaned_L1R',
      'HLT_Photon110_NoHE_Cleaned_L1R_v1',
      'HLT_Photon17_Isol_SC17HE_L1R_v1',
      'HLT_Photon20_Cleaned_L1R',
      'HLT_Photon20_Isol_Cleaned_L1R_v1',
      'HLT_Photon20_NoHE_L1R',
      'HLT_Photon22_SC22HE_L1R_v1',
      'HLT_Photon30_Cleaned_L1R',
      'HLT_Photon40_CaloId_Cleaned_L1R_v1',
      'HLT_Photon40_Isol_Cleaned_L1R_v1',
      'HLT_Photon50_Cleaned_L1R_v1',
      'HLT_Photon50_NoHE_L1R',
      'HLT_Photon70_Cleaned_L1R_v1',
      'HLT_PixelTracks_Multiplicity100',
      'HLT_PixelTracks_Multiplicity70',
      'HLT_PixelTracks_Multiplicity85',
      'HLT_QuadJet15U_v3',
      'HLT_QuadJet20U_v3',
      'HLT_QuadJet25U_v3',
      'HLT_R010U_MR50U',
      'HLT_R030U_MR100U',
      'HLT_R033U_MR100U',
      'HLT_RP025U_MR70U',
      'HLT_RPCBarrelCosmics',
      'HLT_Random',
      'HLT_SingleIsoTau20_Trk15_MET25_v4',
      'HLT_SingleIsoTau35_Trk15_MET25_v4',
      'HLT_StoppedHSCP20_v3',
      'HLT_StoppedHSCP35_v3',
      'HLT_TechTrigHCALNoise',
      'HLT_TrackerCosmics',
      'HLT_ZeroBias',
      'HLT_ZeroBiasPixel_SingleTrack' ),
    hltResults = cms.InputTag( "TriggerResults" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMask = cms.bool( False ),
    daqPartitions = cms.uint32( 1 ),
    throw = cms.bool( True ),
    l1techIgnorePrescales = cms.bool( False )
)
process.hltPreHLTDQMResults = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHLTDQMResultsSmart = cms.EDFilter( "TriggerResultsFilter",
    triggerConditions = cms.vstring( 'HLT_*' ),
    hltResults = cms.InputTag( "TriggerResults" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMask = cms.bool( False ),
    daqPartitions = cms.uint32( 1 ),
    throw = cms.bool( True ),
    l1techIgnorePrescales = cms.bool( False )
)
process.hltPreHLTMON = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
process.hltPreHLTMONSmart = cms.EDFilter( "TriggerResultsFilter",
    triggerConditions = cms.vstring( 'AlCa_EcalEta / 250',
      'AlCa_EcalPhiSym / 250',
      'AlCa_EcalPi0 / 250',
      'AlCa_RPCMuonNoHits / 250',
      'AlCa_RPCMuonNoTriggers / 250',
      'AlCa_RPCMuonNormalisation / 250',
      'HLT_Activity_CSC',
      'HLT_Activity_DT',
      'HLT_Activity_DT_Tuned',
      'HLT_Activity_Ecal_SC17',
      'HLT_Activity_Ecal_SC7',
      'HLT_BTagMu_DiJet10U_v3',
      'HLT_BTagMu_DiJet20U_Mu5_v3',
      'HLT_BTagMu_DiJet20U_v3',
      'HLT_BTagMu_DiJet30U_Mu5_v3',
      'HLT_BTagMu_DiJet30U_v3',
      'HLT_Calibration / 0',
      'HLT_DTErrors',
      'HLT_DiJet20U_Meff180U_v3',
      'HLT_DiJet50U_PT50U_v3',
      'HLT_DiJetAve100U_v3',
      'HLT_DiJetAve140U_v3',
      'HLT_DiJetAve15U_v3',
      'HLT_DiJetAve30U_v3',
      'HLT_DiJetAve50U_v3',
      'HLT_DiJetAve70U_v3',
      'HLT_DoubleEle17_SW_L1R_v1',
      'HLT_DoubleEle4_SW_eeRes_L1R_v2',
      'HLT_DoubleEle5_SW_Upsilon_L1R_v2',
      'HLT_DoubleEle8_SW_HT70U_L1R_v1',
      'HLT_DoubleIsoTau15_OneLeg_Trk5_v4',
      'HLT_DoubleIsoTau15_Trk5_v4',
      'HLT_DoubleJet15U_ForwardBackward_v3',
      'HLT_DoubleJet25U_ForwardBackward_v3',
      'HLT_DoubleJet35U_ForwardBackward_v3',
      'HLT_DoubleMu0',
      'HLT_DoubleMu0_Quarkonium_v1',
      'HLT_DoubleMu3_HT50U_v3',
      'HLT_DoubleMu3_v2',
      'HLT_DoubleMu5_v1',
      'HLT_DoublePhoton17_SingleIsol_L1R_v1',
      'HLT_DoublePhoton22_L1R_v1',
      'HLT_DoublePhoton5_CEP_L1R_v3',
      'HLT_EcalCalibration / 0',
      'HLT_EcalOnly_SumEt160_v3',
      'HLT_Ele10_SW_EleId_HT70U_L1R_v2',
      'HLT_Ele10_SW_HT100U_L1R_v2',
      'HLT_Ele10_SW_HT70U_L1R_v2',
      'HLT_Ele10_SW_L1R_v2',
      'HLT_Ele12_SW_TighterEleId_L1R_v2',
      'HLT_Ele17_SW_Isol_L1R_v2',
      'HLT_Ele17_SW_L1R_v2',
      'HLT_Ele17_SW_TightCaloEleId_Ele8HE_L1R_v2',
      'HLT_Ele17_SW_TighterEleIdIsol_L1R_v3',
      'HLT_Ele22_SW_L1R_v2',
      'HLT_Ele22_SW_TighterCaloIdIsol_L1R_v2',
      'HLT_Ele22_SW_TighterEleId_L1R_v3',
      'HLT_Ele32_SW_TighterEleId_L1R_v2',
      'HLT_ExclDiJet30U_HFAND_v3',
      'HLT_ExclDiJet30U_HFOR_v3',
      'HLT_GlobalRunHPDNoise',
      'HLT_HT100U_v3',
      'HLT_HT130U_v3',
      'HLT_HT140U_J30U_Eta3_v3',
      'HLT_HT150U_Eta3_v3',
      'HLT_HT150U_v3',
      'HLT_HT160U_Eta3_v3',
      'HLT_HT160U_v3',
      'HLT_HT200U_v3',
      'HLT_HT50U_v3',
      'HLT_HcalCalibration / 0',
      'HLT_HcalNZS',
      'HLT_HcalPhiSym',
      'HLT_IsoEle12_PFTau15_v3',
      'HLT_IsoMu11_v4',
      'HLT_IsoMu13_v4',
      'HLT_IsoMu15_v4',
      'HLT_IsoMu17_v4',
      'HLT_IsoMu9_PFTau15_v2',
      'HLT_IsoMu9_v4',
      'HLT_IsoTrackHB_v2',
      'HLT_IsoTrackHE_v3',
      'HLT_Jet100U_v3',
      'HLT_Jet140U_v3',
      'HLT_Jet15U_HcalNoiseFiltered_v3',
      'HLT_Jet15U_v3',
      'HLT_Jet180U_v3',
      'HLT_Jet30U_v3',
      'HLT_Jet50U_v3',
      'HLT_Jet70U_v3',
      'HLT_L1DoubleMuOpen',
      'HLT_L1ETT100',
      'HLT_L1Jet10U',
      'HLT_L1Jet6U',
      'HLT_L1MET20',
      'HLT_L1Mu20',
      'HLT_L1Mu7_v1',
      'HLT_L1MuOpen_AntiBPTX_v2 / 100',
      'HLT_L1MuOpen_DT_v2',
      'HLT_L1MuOpen_v2',
      'HLT_L1SingleEG8',
      'HLT_L1Tech_BSC_HighMultiplicity',
      'HLT_L1Tech_BSC_halo',
      'HLT_L1Tech_BSC_halo_forPhysicsBackground',
      'HLT_L1Tech_BSC_minBias',
      'HLT_L1Tech_BSC_minBias_OR',
      'HLT_L1Tech_HCAL_HF',
      'HLT_L1Tech_RPC_TTU_RBst1_collisions',
      'HLT_L1_BPTX',
      'HLT_L1_BPTX_MinusOnly',
      'HLT_L1_BPTX_PlusOnly',
      'HLT_L1_BptxXOR_BscMinBiasOR',
      'HLT_L2DoubleMu0',
      'HLT_L2DoubleMu20_NoVertex_v1',
      'HLT_L2Mu0_NoVertex',
      'HLT_L2Mu30_v1',
      'HLT_L2Mu7_v1',
      'HLT_LogMonitor',
      'HLT_MET100_v3',
      'HLT_MET120_v3',
      'HLT_MET45_DiJet30U_v3',
      'HLT_MET45_v3',
      'HLT_MET65_CenJet50U_v3',
      'HLT_MET80_CenJet50U_v3',
      'HLT_MinBiasPixel_SingleTrack',
      'HLT_Mu0_TkMu0_OST_Jpsi_Tight_v3',
      'HLT_Mu0_v2',
      'HLT_Mu11',
      'HLT_Mu11_Ele8_v1',
      'HLT_Mu11_PFTau15_v2',
      'HLT_Mu13_v1',
      'HLT_Mu15_v1',
      'HLT_Mu17_v1',
      'HLT_Mu19_v1',
      'HLT_Mu21_v1',
      'HLT_Mu25_v1',
      'HLT_Mu30_NoVertex_v1',
      'HLT_Mu3_Ele8_HT70U_v1',
      'HLT_Mu3_TkMu0_OST_Jpsi_Tight_v3',
      'HLT_Mu3_Track3_Jpsi_v3',
      'HLT_Mu3_Track5_Jpsi_v3',
      'HLT_Mu3_v2',
      'HLT_Mu5',
      'HLT_Mu5_Ele13_v2',
      'HLT_Mu5_Ele17_v2',
      'HLT_Mu5_HT100U_v3',
      'HLT_Mu5_HT70U_v3',
      'HLT_Mu5_Jet50U_v3',
      'HLT_Mu5_Jet70U_v3',
      'HLT_Mu5_L2Mu0',
      'HLT_Mu5_MET45_v3',
      'HLT_Mu5_TkMu0_OST_Jpsi_Tight_v2',
      'HLT_Mu5_Track0_Jpsi_v2',
      'HLT_Mu7',
      'HLT_Mu7_Photon13_Cleaned_L1R_v1',
      'HLT_Mu8_Ele8_v1',
      'HLT_Mu9',
      'HLT_MultiVertex6_v2',
      'HLT_MultiVertex8_L1ETT60_v2',
      'HLT_Photon10_Cleaned_L1R',
      'HLT_Photon110_NoHE_Cleaned_L1R_v1',
      'HLT_Photon17_Isol_SC17HE_L1R_v1',
      'HLT_Photon20_Cleaned_L1R',
      'HLT_Photon20_Isol_Cleaned_L1R_v1',
      'HLT_Photon20_NoHE_L1R',
      'HLT_Photon22_SC22HE_L1R_v1',
      'HLT_Photon30_Cleaned_L1R',
      'HLT_Photon40_CaloId_Cleaned_L1R_v1',
      'HLT_Photon40_Isol_Cleaned_L1R_v1',
      'HLT_Photon50_Cleaned_L1R_v1',
      'HLT_Photon50_NoHE_L1R',
      'HLT_Photon70_Cleaned_L1R_v1',
      'HLT_PixelTracks_Multiplicity100',
      'HLT_PixelTracks_Multiplicity70',
      'HLT_PixelTracks_Multiplicity85',
      'HLT_QuadJet15U_v3',
      'HLT_QuadJet20U_v3',
      'HLT_QuadJet25U_v3',
      'HLT_R010U_MR50U',
      'HLT_R030U_MR100U',
      'HLT_R033U_MR100U',
      'HLT_RP025U_MR70U',
      'HLT_RPCBarrelCosmics',
      'HLT_Random',
      'HLT_SingleIsoTau20_Trk15_MET25_v4',
      'HLT_SingleIsoTau35_Trk15_MET25_v4',
      'HLT_StoppedHSCP20_v3',
      'HLT_StoppedHSCP35_v3',
      'HLT_TechTrigHCALNoise',
      'HLT_TrackerCosmics',
      'HLT_ZeroBias',
      'HLT_ZeroBiasPixel_SingleTrack' ),
    hltResults = cms.InputTag( "TriggerResults" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMask = cms.bool( False ),
    daqPartitions = cms.uint32( 1 ),
    throw = cms.bool( True ),
    l1techIgnorePrescales = cms.bool( False )
)
process.hltPreNanoDST = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)

process.hltOutputA = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputA.root" ),
    fastCloning = cms.untracked.bool( False ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIActivityHF_Coincidence3',
  'HLT_HIActivityHF_Single3',
  'HLT_HIActivityL1A',
  'HLT_HIActivityPixel_SingleTrack',
  'HLT_HIActivityPixels',
  'HLT_HIBptxXOR',
  'HLT_HICentralityVeto',
  'HLT_HIClusterVertexCompatibility',
  'HLT_HIDoublePhoton5_CEP_L1R',
  'HLT_HIJet35U',
  'HLT_HIJet35U_Core',
  'HLT_HIJet50U',
  'HLT_HIJet50U_Core',
  'HLT_HIJet75U',
  'HLT_HIJet75U_Core',
  'HLT_HIJet90U',
  'HLT_HIJet90U_Core',
  'HLT_HIL1Algo_BptxAND',
  'HLT_HIL1Algo_BptxAND_NotBSC2',
  'HLT_HIL1Algo_HFAny',
  'HLT_HIL1Algo_NotBSC2',
  'HLT_HIL1Algo_Unprotected',
  'HLT_HIL1Algo_ZDC',
  'HLT_HIL1DoubleMuOpen',
  'HLT_HIL1DoubleMuOpen_Core',
  'HLT_HIL1ETT30',
  'HLT_HIL1ETT30_Core',
  'HLT_HIL1ETT60',
  'HLT_HIL1ETT60_Core',
  'HLT_HIL1SingleMu3',
  'HLT_HIL1SingleMu3_Core',
  'HLT_HIL1SingleMu5',
  'HLT_HIL1SingleMu5_Core',
  'HLT_HIL1SingleMu7',
  'HLT_HIL1SingleMu7_Core',
  'HLT_HIL1Tech_BSC_HighMultiplicity',
  'HLT_HIL1Tech_BSC_minBias',
  'HLT_HIL1Tech_BSC_minBias_OR',
  'HLT_HIMinBiasBSC',
  'HLT_HIMinBiasBSC_Core',
  'HLT_HIMinBiasBSC_OR',
  'HLT_HIMinBiasBSC_OR_Core',
  'HLT_HIMinBiasHF',
  'HLT_HIMinBiasHF_Core',
  'HLT_HIMinBiasPixel_SingleTrack',
  'HLT_HIMinBiasZDC_Calo',
  'HLT_HIPhoton15',
  'HLT_HIPhoton15_Core',
  'HLT_HIPhoton20',
  'HLT_HIPhoton20_Core',
  'HLT_HIPhoton30',
  'HLT_HIPhoton30_Core',
  'HLT_HIRandom',
  'HLT_HIStoppedHSCP35',
  'HLT_HIUpcEcal',
  'HLT_HIUpcEcal_Core',
  'HLT_HIUpcMu',
  'HLT_HIUpcMu_Core',
  'HLT_HIZeroBias',
  'HLT_HIZeroBiasPixel_SingleTrack',
  'HLT_HIZeroBiasXOR' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputOnlineErrors = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputOnlineErrors.root" ),
    fastCloning = cms.untracked.bool( False ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_LogMonitor' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputExpress = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputExpress.root" ),
    fastCloning = cms.untracked.bool( False ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HICentralityVeto',
  'HLT_HIJet35U',
  'HLT_HIL1DoubleMuOpen',
  'HLT_HIL1Tech_BSC_HighMultiplicity',
  'HLT_HIMinBiasBSC_OR',
  'HLT_HIMinBiasHF',
  'HLT_HIPhoton15' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputEcalCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputEcalCalibration.root" ),
    fastCloning = cms.untracked.bool( False ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_EcalCalibration' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputCalibrationHI = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputCalibrationHI.root" ),
    fastCloning = cms.untracked.bool( False ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_EcalCalibration',
  'HLT_HcalCalibration_HI' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep *_hltHcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputDQM = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputDQM.root" ),
    fastCloning = cms.untracked.bool( False ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_EcalCalibration',
  'HLT_HIActivityL1A',
  'HLT_HICentralityVeto',
  'HLT_HIJet35U_Core',
  'HLT_HIL1DoubleMuOpen_Core',
  'HLT_HIL1SingleMu3',
  'HLT_HIL1Tech_BSC_HighMultiplicity',
  'HLT_HIMinBiasBSC',
  'HLT_HIMinBiasBSC_OR_Core',
  'HLT_HIMinBiasHF',
  'HLT_HIPhoton15_Core',
  'HLT_HcalCalibration_HI',
  'HLT_LogMonitor' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep *_hltDt4DSegments_*_*',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEventWithRefs_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputHLTDQM = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputHLTDQM.root" ),
    fastCloning = cms.untracked.bool( False ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIActivityHF_Coincidence3',
  'HLT_HIActivityHF_Single3',
  'HLT_HIActivityL1A',
  'HLT_HIActivityPixel_SingleTrack',
  'HLT_HIActivityPixels',
  'HLT_HIBptxXOR',
  'HLT_HICentralityVeto',
  'HLT_HIClusterVertexCompatibility',
  'HLT_HIDoublePhoton5_CEP_L1R',
  'HLT_HIJet35U',
  'HLT_HIJet50U',
  'HLT_HIJet75U',
  'HLT_HIJet90U',
  'HLT_HIL1Algo_BptxAND',
  'HLT_HIL1Algo_BptxAND_NotBSC2',
  'HLT_HIL1Algo_HFAny',
  'HLT_HIL1Algo_NotBSC2',
  'HLT_HIL1Algo_Unprotected',
  'HLT_HIL1Algo_ZDC',
  'HLT_HIL1DoubleMuOpen',
  'HLT_HIL1ETT30',
  'HLT_HIL1ETT60',
  'HLT_HIL1SingleMu3',
  'HLT_HIL1SingleMu5',
  'HLT_HIL1SingleMu7',
  'HLT_HIL1Tech_BSC_HighMultiplicity',
  'HLT_HIL1Tech_BSC_minBias',
  'HLT_HIL1Tech_BSC_minBias_OR',
  'HLT_HIMinBiasBSC_OR',
  'HLT_HIMinBiasHF',
  'HLT_HIMinBiasPixel_SingleTrack',
  'HLT_HIMinBiasZDC_Calo',
  'HLT_HIPhoton15',
  'HLT_HIPhoton20',
  'HLT_HIPhoton30',
  'HLT_HIRandom',
  'HLT_HIStoppedHSCP35',
  'HLT_HIUpcEcal',
  'HLT_HIUpcMu',
  'HLT_HIZeroBias',
  'HLT_HIZeroBiasPixel_SingleTrack',
  'HLT_LogMonitor' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep *_hltAlCaEtaRecHitsFilter_*_*',
      'keep *_hltAlCaPhiSymStream_*_*',
      'keep *_hltAlCaPi0RecHitsFilter_*_*',
      'keep *_hltBLifetimeL25AssociatorStartupU_*_*',
      'keep *_hltBLifetimeL25BJetTagsStartupU_*_*',
      'keep *_hltBLifetimeL25JetsStartupU_*_*',
      'keep *_hltBLifetimeL25TagInfosStartupU_*_*',
      'keep *_hltBLifetimeL3AssociatorStartupU_*_*',
      'keep *_hltBLifetimeL3BJetTagsStartupU_*_*',
      'keep *_hltBLifetimeL3JetsStartupU_*_*',
      'keep *_hltBLifetimeL3TagInfosStartupU_*_*',
      'keep *_hltBLifetimeRegionalCtfWithMaterialTracksStartupU_*_*',
      'keep *_hltBSoftMuonL25BJetTagsUByDR_*_*',
      'keep *_hltBSoftMuonL25JetsU_*_*',
      'keep *_hltBSoftMuonL25TagInfosU_*_*',
      'keep *_hltBSoftMuonL3BJetTagsUByDR_*_*',
      'keep *_hltBSoftMuonL3TagInfosU_*_*',
      'keep *_hltCsc2DRecHits_*_*',
      'keep *_hltCscSegments_*_*',
      'keep *_hltDt4DSegments_*_*',
      'keep *_hltFilterL25LeadingTrackPtCutDoubleIsoTau15Trk5_*_*',
      'keep *_hltFilterL25LeadingTrackPtCutSingleIsoTau30Trk5MET20_*_*',
      'keep *_hltFilterL2EcalIsolationDoubleIsoTau15Trk5_*_*',
      'keep *_hltFilterL2EcalIsolationDoubleLooseIsoTau15_*_*',
      'keep *_hltFilterL2EcalIsolationSingleIsoTau30Trk5MET20_*_*',
      'keep *_hltFilterL2EcalIsolationSingleLooseIsoTau20_*_*',
      'keep *_hltFilterL2EtCutDoubleIsoTau15Trk5_*_*',
      'keep *_hltFilterL2EtCutDoubleLooseIsoTau15_*_*',
      'keep *_hltFilterL2EtCutSingleIsoTau30Trk5MET20_*_*',
      'keep *_hltFilterL2EtCutSingleLooseIsoTau20_*_*',
      'keep *_hltFilterL3TrackIsolationDoubleIsoTau15Trk5_*_*',
      'keep *_hltFilterL3TrackIsolationSingleIsoTau30Trk5MET20_*_*',
      'keep *_hltGtDigis_*_*',
      'keep *_hltHITCtfWithMaterialTracksHB8E29_*_*',
      'keep *_hltHITCtfWithMaterialTracksHE8E29_*_*',
      'keep *_hltHITIPTCorrectorHB8E29_*_*',
      'keep *_hltHITIPTCorrectorHE8E29_*_*',
      'keep *_hltHcalDigis_*_*',
      'keep *_hltIconeCentral1Regional_*_*',
      'keep *_hltIconeCentral2Regional_*_*',
      'keep *_hltIconeCentral3Regional_*_*',
      'keep *_hltIconeCentral4Regional_*_*',
      'keep *_hltIconeTau1Regional_*_*',
      'keep *_hltIconeTau2Regional_*_*',
      'keep *_hltIconeTau3Regional_*_*',
      'keep *_hltIconeTau4Regional_*_*',
      'keep *_hltIsolPixelTrackProdHB8E29_*_*',
      'keep *_hltIsolPixelTrackProdHE8E29_*_*',
      'keep *_hltIterativeCone5CaloJets_*_*',
      'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*',
      'keep *_hltL1IsoRecoEcalCandidate_*_*',
      'keep *_hltL1IsoSiStripElectronPixelSeeds_*_*',
      'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*',
      'keep *_hltL1IsolatedElectronHcalIsol_*_*',
      'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*',
      'keep *_hltL1NonIsoRecoEcalCandidate_*_*',
      'keep *_hltL1NonIsoSiStripElectronPixelSeeds_*_*',
      'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*',
      'keep *_hltL1NonIsolatedElectronHcalIsol_*_*',
      'keep *_hltL1extraParticles_*_*',
      'keep *_hltL1sDoubleLooseIsoTau15_*_*',
      'keep *_hltL1sSingleLooseIsoTau20_*_*',
      'keep *_hltL25TauConeIsolation_*_*',
      'keep *_hltL25TauCtfWithMaterialTracks_*_*',
      'keep *_hltL25TauJetTracksAssociator_*_*',
      'keep *_hltL25TauLeadingTrackPtCutSelector_*_*',
      'keep *_hltL2MuonCandidates_*_*',
      'keep *_hltL2MuonIsolations_*_*',
      'keep *_hltL2MuonSeeds_*_*',
      'keep *_hltL2Muons_*_*',
      'keep *_hltL2TauJets_*_*',
      'keep *_hltL2TauNarrowConeIsolationProducer_*_*',
      'keep *_hltL2TauRelaxingIsolationSelector_*_*',
      'keep *_hltL3MuonCandidates_*_*',
      'keep *_hltL3MuonIsolations_*_*',
      'keep *_hltL3MuonsIOHit_*_*',
      'keep *_hltL3MuonsLinksCombination_*_*',
      'keep *_hltL3MuonsOIHit_*_*',
      'keep *_hltL3MuonsOIState_*_*',
      'keep *_hltL3Muons_*_*',
      'keep *_hltL3TauConeIsolation_*_*',
      'keep *_hltL3TauCtfWithMaterialTracks_*_*',
      'keep *_hltL3TauIsolationSelector_*_*',
      'keep *_hltL3TauJetTracksAssociator_*_*',
      'keep *_hltL3TkFromL2OICombination_*_*',
      'keep *_hltL3TkTracksFromL2IOHit_*_*',
      'keep *_hltL3TkTracksFromL2OIHit_*_*',
      'keep *_hltL3TkTracksFromL2OIState_*_*',
      'keep *_hltL3TrackCandidateFromL2IOHit_*_*',
      'keep *_hltL3TrackCandidateFromL2OIHit_*_*',
      'keep *_hltL3TrackCandidateFromL2OIState_*_*',
      'keep *_hltL3TrajSeedIOHit_*_*',
      'keep *_hltL3TrajSeedOIHit_*_*',
      'keep *_hltL3TrajSeedOIState_*_*',
      'keep *_hltL3TrajectorySeed_*_*',
      'keep *_hltMCJetCorJetIcone5HF07_*_*',
      'keep *_hltMet_*_*',
      'keep *_hltMuTrackJpsiCtfTrackCands_*_*',
      'keep *_hltMuTrackJpsiPixelTrackCands_*_*',
      'keep *_hltMuonCSCDigis_*_*',
      'keep *_hltMuonRPCDigis_*_*',
      'keep *_hltOfflineBeamSpot_*_*',
      'keep *_hltPixelMatchLargeWindowElectronsL1Iso_*_*',
      'keep *_hltPixelMatchLargeWindowElectronsL1NonIso_*_*',
      'keep *_hltPixelTracks_*_*',
      'keep *_hltRpcRecHits_*_*',
      'keep *_hltSiPixelClusters_*_*',
      'keep *_hltSiStripRawToClustersFacility_*_*',
      'keep *_hltTowerMakerForMuons_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEventWithRefs_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputHLTDQMResults = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputHLTDQMResults.root" ),
    fastCloning = cms.untracked.bool( False ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLTriggerFinalPath' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep LumiScalerss_*_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputHLTMON = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputHLTMON.root" ),
    fastCloning = cms.untracked.bool( False ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIActivityHF_Coincidence3',
  'HLT_HIActivityHF_Single3',
  'HLT_HIActivityL1A',
  'HLT_HIActivityPixel_SingleTrack',
  'HLT_HIActivityPixels',
  'HLT_HIBptxXOR',
  'HLT_HICentralityVeto',
  'HLT_HIClusterVertexCompatibility',
  'HLT_HIDoublePhoton5_CEP_L1R',
  'HLT_HIJet35U',
  'HLT_HIJet50U',
  'HLT_HIJet75U',
  'HLT_HIJet90U',
  'HLT_HIL1Algo_BptxAND',
  'HLT_HIL1Algo_BptxAND_NotBSC2',
  'HLT_HIL1Algo_HFAny',
  'HLT_HIL1Algo_NotBSC2',
  'HLT_HIL1Algo_Unprotected',
  'HLT_HIL1Algo_ZDC',
  'HLT_HIL1DoubleMuOpen',
  'HLT_HIL1ETT30',
  'HLT_HIL1ETT60',
  'HLT_HIL1SingleMu3',
  'HLT_HIL1SingleMu5',
  'HLT_HIL1SingleMu7',
  'HLT_HIL1Tech_BSC_HighMultiplicity',
  'HLT_HIL1Tech_BSC_minBias',
  'HLT_HIL1Tech_BSC_minBias_OR',
  'HLT_HIMinBiasBSC_OR',
  'HLT_HIMinBiasHF',
  'HLT_HIMinBiasPixel_SingleTrack',
  'HLT_HIMinBiasZDC_Calo',
  'HLT_HIPhoton15',
  'HLT_HIPhoton20',
  'HLT_HIPhoton30',
  'HLT_HIRandom',
  'HLT_HIStoppedHSCP35',
  'HLT_HIUpcEcal',
  'HLT_HIUpcMu',
  'HLT_HIZeroBias',
  'HLT_HIZeroBiasPixel_SingleTrack',
  'HLT_LogMonitor' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep *_hltAlCaEtaRecHitsFilter_*_*',
      'keep *_hltAlCaPi0RecHitsFilter_*_*',
      'keep *_hltBLifetimeL25AssociatorStartupU_*_*',
      'keep *_hltBLifetimeL25BJetTagsStartupU_*_*',
      'keep *_hltBLifetimeL25JetsStartupU_*_*',
      'keep *_hltBLifetimeL25TagInfosStartupU_*_*',
      'keep *_hltBLifetimeL3AssociatorStartupU_*_*',
      'keep *_hltBLifetimeL3BJetTagsStartupU_*_*',
      'keep *_hltBLifetimeL3JetsStartupU_*_*',
      'keep *_hltBLifetimeL3TagInfosStartupU_*_*',
      'keep *_hltBLifetimeRegionalCtfWithMaterialTracksStartupU_*_*',
      'keep *_hltBSoftMuonL25BJetTagsUByDR_*_*',
      'keep *_hltBSoftMuonL25JetsU_*_*',
      'keep *_hltBSoftMuonL25TagInfosU_*_*',
      'keep *_hltBSoftMuonL3BJetTagsUByDR_*_*',
      'keep *_hltBSoftMuonL3BJetTagsUByPt_*_*',
      'keep *_hltBSoftMuonL3TagInfosU_*_*',
      'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*',
      'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*',
      'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*',
      'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*',
      'keep *_hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated_*_*',
      'keep *_hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated_*_*',
      'keep *_hltCscSegments_*_*',
      'keep *_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*',
      'keep *_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*',
      'keep *_hltDt1DRecHits_*_*',
      'keep *_hltDt4DSegments_*_*',
      'keep *_hltFilterL25LeadingTrackPtCutDoubleIsoTau15Trk5_*_*',
      'keep *_hltFilterL25LeadingTrackPtCutSingleIsoTau30Trk5MET20_*_*',
      'keep *_hltFilterL2EcalIsolationDoubleIsoTau15Trk5_*_*',
      'keep *_hltFilterL2EcalIsolationDoubleLooseIsoTau15_*_*',
      'keep *_hltFilterL2EcalIsolationSingleIsoTau30Trk5MET20_*_*',
      'keep *_hltFilterL2EcalIsolationSingleLooseIsoTau20_*_*',
      'keep *_hltFilterL2EtCutDoubleIsoTau15Trk5_*_*',
      'keep *_hltFilterL2EtCutDoubleLooseIsoTau15_*_*',
      'keep *_hltFilterL2EtCutSingleIsoTau30Trk5MET20_*_*',
      'keep *_hltFilterL2EtCutSingleLooseIsoTau20_*_*',
      'keep *_hltFilterL3TrackIsolationDoubleIsoTau15Trk5_*_*',
      'keep *_hltFilterL3TrackIsolationSingleIsoTau30Trk5MET20_*_*',
      'keep *_hltGctDigis_*_*',
      'keep *_hltGtDigis_*_*',
      'keep *_hltHoreco_*_*',
      'keep *_hltIconeCentral1Regional_*_*',
      'keep *_hltIconeCentral2Regional_*_*',
      'keep *_hltIconeCentral3Regional_*_*',
      'keep *_hltIconeCentral4Regional_*_*',
      'keep *_hltIconeTau1Regional_*_*',
      'keep *_hltIconeTau2Regional_*_*',
      'keep *_hltIconeTau3Regional_*_*',
      'keep *_hltIconeTau4Regional_*_*',
      'keep *_hltL1GtObjectMap_*_*',
      'keep *_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*',
      'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*',
      'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*',
      'keep *_hltL1IsoHLTClusterShape_*_*',
      'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*',
      'keep *_hltL1IsoPhotonHollowTrackIsol_*_*',
      'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*',
      'keep *_hltL1IsolatedPhotonEcalIsol_*_*',
      'keep *_hltL1IsolatedPhotonHcalIsol_*_*',
      'keep *_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*',
      'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*',
      'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*',
      'keep *_hltL1NonIsoHLTClusterShape_*_*',
      'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*',
      'keep *_hltL1NonIsoPhotonHollowTrackIsol_*_*',
      'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*',
      'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*',
      'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*',
      'keep *_hltL1extraParticles_*_*',
      'keep *_hltL1sDoubleLooseIsoTau15_*_*',
      'keep *_hltL1sSingleLooseIsoTau20_*_*',
      'keep *_hltL25TauConeIsolation_*_*',
      'keep *_hltL25TauCtfWithMaterialTracks_*_*',
      'keep *_hltL25TauJetTracksAssociator_*_*',
      'keep *_hltL25TauLeadingTrackPtCutSelector_*_*',
      'keep *_hltL2MuonCandidatesNoVtx_*_*',
      'keep *_hltL2MuonCandidates_*_*',
      'keep *_hltL2MuonIsolations_*_*',
      'keep *_hltL2MuonSeeds_*_*',
      'keep *_hltL2Muons_*_*',
      'keep *_hltL2TauJets_*_*',
      'keep *_hltL2TauNarrowConeIsolationProducer_*_*',
      'keep *_hltL2TauRelaxingIsolationSelector_*_*',
      'keep *_hltL3MuonCandidatesNoVtx_*_*',
      'keep *_hltL3MuonCandidates_*_*',
      'keep *_hltL3MuonIsolations_*_*',
      'keep *_hltL3MuonsIOHit_*_*',
      'keep *_hltL3MuonsLinksCombination_*_*',
      'keep *_hltL3MuonsNoVtx_*_*',
      'keep *_hltL3MuonsOIHit_*_*',
      'keep *_hltL3MuonsOIState_*_*',
      'keep *_hltL3Muons_*_*',
      'keep *_hltL3TauConeIsolation_*_*',
      'keep *_hltL3TauCtfWithMaterialTracks_*_*',
      'keep *_hltL3TauIsolationSelector_*_*',
      'keep *_hltL3TauJetTracksAssociator_*_*',
      'keep *_hltL3TkFromL2OICombination_*_*',
      'keep *_hltL3TkTracksFromL2IOHit_*_*',
      'keep *_hltL3TkTracksFromL2OIHit_*_*',
      'keep *_hltL3TkTracksFromL2OIState_*_*',
      'keep *_hltL3TkTracksFromL2_*_*',
      'keep *_hltL3TrackCandidateFromL2IOHit_*_*',
      'keep *_hltL3TrackCandidateFromL2OIHit_*_*',
      'keep *_hltL3TrackCandidateFromL2OIState_*_*',
      'keep *_hltL3TrackCandidateFromL2_*_*',
      'keep *_hltL3TrajSeedIOHit_*_*',
      'keep *_hltL3TrajSeedOIHit_*_*',
      'keep *_hltL3TrajSeedOIState_*_*',
      'keep *_hltL3TrajectorySeedNoVtx_*_*',
      'keep *_hltL3TrajectorySeed_*_*',
      'keep *_hltMet_*_*',
      'keep *_hltMuTrackJpsiCtfTrackCands_*_*',
      'keep *_hltMuTrackJpsiCtfTracks_*_*',
      'keep *_hltMuTrackJpsiPixelTrackCands_*_*',
      'keep *_hltMuTrackJpsiPixelTrackSelector_*_*',
      'keep *_hltMuTrackJpsiTrackSeeds_*_*',
      'keep *_hltMuonRPCDigis_*_*',
      'keep *_hltOfflineBeamSpot_*_*',
      'keep *_hltPixelMatchElectronsL1Iso_*_*',
      'keep *_hltPixelMatchElectronsL1NonIso_*_*',
      'keep *_hltPixelMatchLargeWindowElectronsL1Iso_*_*',
      'keep *_hltPixelMatchLargeWindowElectronsL1NonIso_*_*',
      'keep *_hltPixelTracks_*_*',
      'keep *_hltPixelVertices_*_*',
      'keep *_hltRpcRecHits_*_*',
      'keep *_hltSiPixelRecHits_*_*',
      'keep *_hltSiStripClusters_*_*',
      'keep *_hltSiStripRawToClustersFacility_*_*',
      'keep *_hltTowerMakerForAll_*_*',
      'keep *_hltTowerMakerForMuons_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep SiPixelClusteredmNewDetSetVector_hltSiPixelClusters_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEventWithRefs_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputNanoDST = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputNanoDST.root" ),
    fastCloning = cms.untracked.bool( False ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLTriggerFinalPath' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep *_hltFEDSelector_*_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep L1MuGMTReadoutCollection_hltGtDigis_*_*',
      'keep edmTriggerResults_*_*_*' )
)

process.HLTEndSequence = cms.Sequence( process.hltBoolEnd )
process.HLTL1UnpackerSequence = cms.Sequence( process.hltGtDigis + process.hltGctDigis + process.hltL1GtObjectMap + process.hltL1extraParticles )
process.HLTBeamSpot = cms.Sequence( process.hltScalersRawToDigi + process.hltOnlineBeamSpot + process.hltOfflineBeamSpot )
process.HLTBeginSequence = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.HLTBeamSpot )
process.HLTDoHILocalPixelSequence = cms.Sequence( process.hltSiPixelDigis + process.hltHISiPixelClusters + process.hltHISiPixelRecHits )
process.HLTPixelTrackingForHITrackTrigger = cms.Sequence( process.hltHIPixelClusterVertices + process.hltPixelTracksForHITrackTrigger + process.hltPixelCandsForHITrackTrigger )
process.HLTBeginSequenceBPTX = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.hltBPTXCoincidence + process.HLTBeamSpot )
process.HLTDoLocalHcalSequence = cms.Sequence( process.hltHcalDigis + process.hltHbhereco + process.hltHfreco + process.hltHoreco )
process.HLTDoCaloSequence = cms.Sequence( process.hltEcalRawToRecHitFacility + process.hltEcalRegionalRestFEDs + process.hltEcalRecHitAll + process.HLTDoLocalHcalSequence + process.hltTowerMakerForAll )
process.HLTDoHIEcalClusSequence = cms.Sequence( process.hltIslandBasicClustersHI + process.hltIslandSuperClustersHI + process.hltCorrectedIslandBarrelSuperClustersHI + process.hltCorrectedIslandEndcapSuperClustersHI + process.hltRecoHIEcalCandidate )
process.HLTDoRegionalEgammaEcalSequence = cms.Sequence( process.hltESRawToRecHitFacility + process.hltEcalRawToRecHitFacility + process.hltEcalRegionalEgammaFEDs + process.hltEcalRegionalEgammaRecHit + process.hltESRegionalEgammaRecHit )
process.HLTMulti5x5SuperClusterL1Isolated = cms.Sequence( process.hltMulti5x5BasicClustersL1Isolated + process.hltMulti5x5SuperClustersL1Isolated + process.hltMulti5x5EndcapSuperClustersWithPreshowerL1Isolated + process.hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated )
process.HLTL1IsolatedEcalClustersSequence = cms.Sequence( process.hltHybridSuperClustersL1Isolated + process.hltCorrectedHybridSuperClustersL1Isolated + process.HLTMulti5x5SuperClusterL1Isolated )
process.HLTMulti5x5SuperClusterL1NonIsolated = cms.Sequence( process.hltMulti5x5BasicClustersL1NonIsolated + process.hltMulti5x5SuperClustersL1NonIsolated + process.hltMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated + process.hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolatedTemp + process.hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated )
process.HLTL1NonIsolatedEcalClustersSequence = cms.Sequence( process.hltHybridSuperClustersL1NonIsolated + process.hltCorrectedHybridSuperClustersL1NonIsolatedTemp + process.hltCorrectedHybridSuperClustersL1NonIsolated + process.HLTMulti5x5SuperClusterL1NonIsolated )
process.HLTHIDoublePhotonEt5Sequence = cms.Sequence( process.HLTDoRegionalEgammaEcalSequence + process.HLTL1IsolatedEcalClustersSequence + process.HLTL1NonIsolatedEcalClustersSequence + process.hltL1IsoRecoEcalCandidate + process.hltL1NonIsoRecoEcalCandidate + process.hltHIDoublePhotonEt5L1MatchFilterRegional + process.hltHIDoublePhotonEt5EtPhiFilter + process.hltL1IsolatedPhotonEcalIsol + process.hltL1NonIsolatedPhotonEcalIsol + process.hltHIDoublePhotonEt5EcalIsolFilter + process.HLTDoLocalHcalSequence + process.hltL1IsolatedPhotonHcalForHE + process.hltL1NonIsolatedPhotonHcalForHE + process.hltHIDoublePhotonEt5HEFilter )
process.HLTDoHIJetRecoSequence = cms.Sequence( process.HLTDoCaloSequence + process.hltIterativeCone5PileupSubtractionCaloJets )
process.HLTBeginSequenceAntiBPTX = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.hltBPTXAntiCoincidence + process.HLTBeamSpot )

process.HLTriggerFirstPath = cms.Path( process.hltGetRaw + process.hltBoolFalse )
process.HLT_LogMonitor = cms.Path( process.hltGtDigis + process.hltPreLogMonitor + process.hltLogMonitorFilter + process.HLTEndSequence )
process.HLT_EcalCalibration = cms.Path( process.hltCalibrationEventsFilter + process.hltGtDigis + process.hltPreEcalCalibration + process.hltEcalCalibrationRaw + process.HLTEndSequence )
process.HLT_HIZeroBias = cms.Path( process.HLTBeginSequence + process.hltL1sHIZeroBias + process.hltPreHIZeroBias + process.HLTEndSequence )
process.HLT_HIZeroBiasXOR = cms.Path( process.HLTBeginSequence + process.hltL1sHIZeroBiasXORonly + process.hltPreHIZeroBiasXOR + process.HLTEndSequence )
process.HLT_HIZeroBiasPixel_SingleTrack = cms.Path( process.HLTBeginSequence + process.hltL1sHIZeroBiasXOR + process.hltPreHIZeroBiasPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter1 + process.HLTEndSequence )
process.HLT_HIMinBiasBSC = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasBSC + process.hltPreHIMinBiasBSC + process.HLTEndSequence )
process.HLT_HIMinBiasBSC_Core = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasBSC + process.hltPreHIMinBiasBSCCore + process.HLTEndSequence )
process.HLT_HIMinBiasBSC_OR = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasBSCOR + process.hltPreHIMinBiasBSCOR + process.HLTEndSequence )
process.HLT_HIMinBiasBSC_OR_Core = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasBSCOR + process.hltPreHIMinBiasBSCORCore + process.HLTEndSequence )
process.HLT_HIMinBiasHF = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasHF + process.hltPreHIMinBiasHF + process.HLTEndSequence )
process.HLT_HIMinBiasHF_Core = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasHF + process.hltPreHIMinBiasHFCore + process.HLTEndSequence )
process.HLT_HIMinBiasPixel_SingleTrack = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasBSCOR + process.hltPreHIMinBiasPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter1 + process.HLTEndSequence )
process.HLT_HIMinBiasZDC_Calo = cms.Path( process.HLTBeginSequence + process.hltL1sHIMinBiasZDC + process.hltPreHIMinBiasZDC + process.HLTEndSequence )
process.HLT_HIBptxXOR = cms.Path( process.HLTBeginSequence + process.hltL1sHIBptxXOR + process.hltPreHIBptxXOR + process.HLTEndSequence )
process.HLT_HIL1Algo_BptxAND = cms.Path( process.HLTBeginSequence + process.hltL1sHIL1AlgoBptxAND + process.hltPreHIL1AlgoBptxAND + process.HLTEndSequence )
process.HLT_HIL1Algo_BptxAND_NotBSC2 = cms.Path( process.HLTBeginSequence + process.hltL1sHIL1AlgoBptxANDNotBSC2 + process.hltPreHIL1AlgoBptxANDNotBSC2 + process.HLTEndSequence )
process.HLT_HIL1Algo_HFAny = cms.Path( process.HLTBeginSequence + process.hltL1sHIL1HFtech + process.hltPreHIL1HFTech + process.HLTEndSequence )
process.HLT_HIL1Algo_NotBSC2 = cms.Path( process.HLTBeginSequence + process.hltL1sHIL1AlgoNotBSC2 + process.hltPreHIL1AlgoNotBSC2 + process.HLTEndSequence )
process.HLT_HIL1Algo_Unprotected = cms.Path( process.HLTBeginSequence + process.hltL1sHIL1AlgoUnprotected + process.hltPreHIL1AlgoUnprotected + process.HLTEndSequence )
process.HLT_HIL1Algo_ZDC = cms.Path( process.HLTBeginSequence + process.hltL1sHIL1AlgoZDC + process.hltPreHIL1AlgoZDC + process.HLTEndSequence )
process.HLT_HIL1SingleMu3 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIL1SingleMu3 + process.hltPreHIL1SingleMu3 + process.HLTEndSequence )
process.HLT_HIL1SingleMu3_Core = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIL1SingleMu3 + process.hltPreHIL1SingleMu3Core + process.HLTEndSequence )
process.HLT_HIL1SingleMu5 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIL1SingleMu5 + process.hltPreHIL1SingleMu5 + process.HLTEndSequence )
process.HLT_HIL1SingleMu5_Core = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIL1SingleMu5 + process.hltPreHIL1SingleMu5Core + process.HLTEndSequence )
process.HLT_HIL1SingleMu7 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIL1SingleMu7 + process.hltPreHIL1SingleMu7 + process.HLTEndSequence )
process.HLT_HIL1SingleMu7_Core = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIL1SingleMu7 + process.hltPreHIL1SingleMu7Core + process.HLTEndSequence )
process.HLT_HIL1DoubleMuOpen = cms.Path( process.HLTBeginSequence + process.hltL1sHIL1DoubleMuOpen + process.hltPreHIL1DoubleMuOpen + process.hltHIDoubleMuLevel1PathL1OpenFiltered + process.HLTEndSequence )
process.HLT_HIL1DoubleMuOpen_Core = cms.Path( process.HLTBeginSequence + process.hltL1sHIL1DoubleMuOpen + process.hltPreHIL1DoubleMuOpenCore + process.hltHIDoubleMuLevel1PathL1OpenFiltered + process.HLTEndSequence )
process.HLT_HIL1ETT30 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIL1ETT30 + process.hltPreHIL1ETT30 + process.HLTEndSequence )
process.HLT_HIL1ETT30_Core = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIL1ETT30 + process.hltPreHIL1ETT30Core + process.HLTEndSequence )
process.HLT_HIL1ETT60 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIL1ETT60 + process.hltPreHIL1ETT60 + process.HLTEndSequence )
process.HLT_HIL1ETT60_Core = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIL1ETT60 + process.hltPreHIL1ETT60Core + process.HLTEndSequence )
process.HLT_HIL1Tech_BSC_minBias = cms.Path( process.HLTBeginSequence + process.hltL1sHIL1TechBSCminBias + process.hltPreHIL1TechBSCminBias + process.HLTEndSequence )
process.HLT_HIL1Tech_BSC_minBias_OR = cms.Path( process.HLTBeginSequence + process.hltL1sHIL1TechBSCminBiasOR + process.hltPreHIL1TechBSCminBiasOR + process.HLTEndSequence )
process.HLT_HIL1Tech_BSC_HighMultiplicity = cms.Path( process.HLTBeginSequence + process.hltL1sHighMultiplicityBSC + process.hltPreHIL1TechBSCHighMultiplicity + process.HLTEndSequence )
process.HLT_HIUpcEcal = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIUpcEcal + process.hltPreHIUpcEcal + process.HLTEndSequence )
process.HLT_HIUpcEcal_Core = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIUpcEcal + process.hltPreHIUpcEcalCore + process.HLTEndSequence )
process.HLT_HIUpcMu = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIUpcMu + process.hltPreHIUpcMu + process.HLTEndSequence )
process.HLT_HIUpcMu_Core = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIUpcMu + process.hltPreHIUpcMuCore + process.HLTEndSequence )
process.HLT_HIPhoton15 = cms.Path( process.HLTBeginSequence + process.hltL1sHIPhoton15 + process.hltPreHIPhoton15 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusSequence + process.hltHIPhoton15 + process.HLTEndSequence )
process.HLT_HIPhoton15_Core = cms.Path( process.HLTBeginSequence + process.hltL1sHIPhoton15 + process.hltPreHIPhoton15Core + process.HLTDoCaloSequence + process.HLTDoHIEcalClusSequence + process.hltHIPhoton15 + process.HLTEndSequence )
process.HLT_HIPhoton20 = cms.Path( process.HLTBeginSequence + process.hltL1sHIPhoton20 + process.hltPreHIPhoton20 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusSequence + process.hltHIPhoton20 + process.HLTEndSequence )
process.HLT_HIPhoton20_Core = cms.Path( process.HLTBeginSequence + process.hltL1sHIPhoton20 + process.hltPreHIPhoton20Core + process.HLTDoCaloSequence + process.HLTDoHIEcalClusSequence + process.hltHIPhoton20 + process.HLTEndSequence )
process.HLT_HIPhoton30 = cms.Path( process.HLTBeginSequence + process.hltL1sHIPhoton30 + process.hltPreHIPhoton30 + process.HLTDoCaloSequence + process.HLTDoHIEcalClusSequence + process.hltHIPhoton30 + process.HLTEndSequence )
process.HLT_HIPhoton30_Core = cms.Path( process.HLTBeginSequence + process.hltL1sHIPhoton30 + process.hltPreHIPhoton30Core + process.HLTDoCaloSequence + process.HLTDoHIEcalClusSequence + process.hltHIPhoton30 + process.HLTEndSequence )
process.HLT_HIDoublePhoton5_CEP_L1R = cms.Path( process.HLTBeginSequence + process.hltL1sHIL1DoubleEG5 + process.hltPreHIDoublePhoton5CEPL1R + process.HLTHIDoublePhotonEt5Sequence + process.hltTowerMakerForHcal + process.hltHcalTowerFilter + process.HLTEndSequence )
process.HLT_HIJet35U = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIJet35U + process.hltPreHIJet35U + process.HLTDoHIJetRecoSequence + process.hltHI1jet35U + process.HLTEndSequence )
process.HLT_HIJet35U_Core = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIJet35U + process.hltPreHIJet35UCore + process.HLTDoHIJetRecoSequence + process.hltHI1jet35U + process.HLTEndSequence )
process.HLT_HIJet50U = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIJet50U + process.hltPreHIJet50U + process.HLTDoHIJetRecoSequence + process.hltHI1jet50U + process.HLTEndSequence )
process.HLT_HIJet50U_Core = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIJet50U + process.hltPreHIJet50UCore + process.HLTDoHIJetRecoSequence + process.hltHI1jet50U + process.HLTEndSequence )
process.HLT_HIJet75U = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIJet75U + process.hltPreHIJet75U + process.HLTDoHIJetRecoSequence + process.hltHI1jet75U + process.HLTEndSequence )
process.HLT_HIJet75U_Core = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIJet75U + process.hltPreHIJet75UCore + process.HLTDoHIJetRecoSequence + process.hltHI1jet75U + process.HLTEndSequence )
process.HLT_HIJet90U = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIJet90U + process.hltPreHIJet90U + process.HLTDoHIJetRecoSequence + process.hltHI1jet90U + process.HLTEndSequence )
process.HLT_HIJet90U_Core = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIJet90U + process.hltPreHIJet90UCopy + process.HLTDoHIJetRecoSequence + process.hltHI1jet90U + process.HLTEndSequence )
process.HLT_HIStoppedHSCP35 = cms.Path( process.HLTBeginSequenceAntiBPTX + process.hltL1sHIL1SingleJet20 + process.hltPreHIStoppedHSCP35 + process.hltHcalDigis + process.hltHbhereco + process.hltStoppedHSCPHpdFilter + process.hltStoppedHSCPTowerMakerForAll + process.hltStoppedHSCPIterativeCone5CaloJets + process.hltStoppedHSCP1CaloJetEnergy35 + process.HLTEndSequence )
process.HLT_HIActivityL1A = cms.Path( process.HLTBeginSequence + process.hltL1sHIGlobalDecision + process.hltPreHIActivityL1A + process.HLTEndSequence )
process.HLT_HIActivityPixels = cms.Path( process.HLTBeginSequence + process.hltL1sHIGlobalDecision + process.hltPreHIActivityPixels + process.HLTDoHILocalPixelSequence + process.hltHIPixelActivityFilter + process.HLTEndSequence )
process.HLT_HIActivityPixel_SingleTrack = cms.Path( process.HLTBeginSequence + process.hltL1sHIGlobalDecision + process.hltPreHIActivityPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter1 + process.HLTEndSequence )
process.HLT_HIActivityHF_Single3 = cms.Path( process.HLTBeginSequence + process.hltL1sHIGlobalDecision + process.hltPreHIActivityHFSingle3 + process.hltHcalDigis + process.hltHfreco + process.hltHcalSimpleRecHitFilter + process.HLTEndSequence )
process.HLT_HIActivityHF_Coincidence3 = cms.Path( process.HLTBeginSequence + process.hltL1sHIGlobalDecision + process.hltPreHIActivityHFCoincidence3 + process.hltHcalDigis + process.hltHfreco + process.hltHcalSimpleRecHitFilterCoincidence + process.HLTEndSequence )
process.HLT_HIClusterVertexCompatibility = cms.Path( process.HLTBeginSequence + process.hltL1sHIGlobalDecision + process.hltPreHIClusterVertexCompatibility + process.HLTDoHILocalPixelSequence + process.hltHIPixelClusterShapeFilter + process.HLTEndSequence )
process.HLT_HICentralityVeto = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sHIMinBiasBSC + process.hltPreHICentralityVeto + process.HLTDoHILocalPixelSequence + process.hltPixelActivityFilterCentralityVeto + process.HLTEndSequence )
process.HLT_HIRandom = cms.Path( process.hltRandomEventsFilter + process.HLTL1UnpackerSequence + process.hltPreHIRandom + process.HLTEndSequence )
process.HLT_HcalCalibration_HI = cms.Path( process.hltCalibrationEventsFilter + process.hltGtDigis + process.hltPreHIHcalCalibration + process.hltHcalCalibTypeFilter + process.hltHcalCalibrationRaw + process.HLTEndSequence )
process.HLTriggerFinalPath = cms.Path( process.hltGtDigis + process.hltFEDSelector + process.hltTriggerSummaryAOD + process.hltTriggerSummaryRAW + process.hltBoolTrue )
process.HLTAnalyzerEndpath = cms.EndPath( process.hltL1GtTrigReport + process.hltTrigReport )
process.HLTOutput = cms.EndPath( process.hltOutputA + process.hltOutputOnlineErrors )
process.ExpressOutput = cms.EndPath( process.hltPreExpress + process.hltPreExpressSmart + process.hltOutputExpress )
process.AlCaOutput = cms.EndPath( process.hltOutputEcalCalibration )
process.AlCaHIOutput = cms.EndPath( process.hltOutputCalibrationHI )
process.DQMOutput = cms.EndPath( process.hltDQML1Scalers + process.hltDQML1SeedLogicScalers + process.hltDQMHLTScalers + process.hltPreDQM + process.hltPreDQMSmart + process.hltOutputDQM )
process.HLTDQMOutput = cms.EndPath( process.hltPreHLTDQM + process.hltPreHLTDQMSmart + process.hltOutputHLTDQM )
process.HLTDQMResultsOutput = cms.EndPath( process.hltPreHLTDQMResults + process.hltPreHLTDQMResultsSmart + process.hltOutputHLTDQMResults )
process.HLTMONOutput = cms.EndPath( process.hltPreHLTMON + process.hltPreHLTMONSmart + process.hltOutputHLTMON )
process.NanoDSTOutput = cms.EndPath( process.hltPreNanoDST + process.hltOutputNanoDST )


# override the preshower baseline setting for MC
if 'ESUnpackerWorkerESProducer' in process.__dict__:
    process.ESUnpackerWorkerESProducer.RHAlgo.ESBaseline = 1000

# HIon paths in smart prescalers
if 'hltPreHLTDQMSmart' in process.__dict__:
    process.hltPreHLTDQMSmart.throw = cms.bool( False )
if 'hltPreHLTMONSmart' in process.__dict__:
    process.hltPreHLTMONSmart.throw = cms.bool( False )
if 'hltPreExpressSmart' in process.__dict__:
    process.hltPreExpressSmart.throw = cms.bool( False )
if 'hltPreDQMSmart' in process.__dict__:
    process.hltPreDQMSmart.throw = cms.bool( False )

# remove HLT prescales
if 'PrescaleService' in process.__dict__:
    process.PrescaleService.lvl1DefaultLabel = cms.untracked.string( '0' )
    process.PrescaleService.lvl1Labels = cms.vstring( '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' )
    process.PrescaleService.prescaleTable = cms.VPSet( )

# set process name
process.setName_('HLTHIon')

# add global options
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True )
)

# override the GlobalTag connection string and pfnPrefix
if 'GlobalTag' in process.__dict__:
    process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
    process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
    process.GlobalTag.globaltag = 'START39_V4HI::All'

# adapt HLT modules to the correct process name
if 'hltTrigReport' in process.__dict__:
    process.hltTrigReport.HLTriggerResults       = cms.InputTag( 'TriggerResults','',process.name_() )

if 'hltDQMHLTScalers' in process.__dict__:
    process.hltDQMHLTScalers.triggerResults      = cms.InputTag( 'TriggerResults','',process.name_() )

if 'hltPreExpressSmart' in process.__dict__:
    process.hltPreExpressSmart.TriggerResultsTag = cms.InputTag( 'TriggerResults','',process.name_() )

if 'hltPreHLTMONSmart' in process.__dict__:
    process.hltPreHLTMONSmart.TriggerResultsTag  = cms.InputTag( 'TriggerResults','',process.name_() )

if 'hltPreDQMSmart' in process.__dict__:
    process.hltPreDQMSmart.TriggerResultsTag     = cms.InputTag( 'TriggerResults','',process.name_() )

if 'hltDQML1SeedLogicScalers' in process.__dict__:
    process.hltDQML1SeedLogicScalers.processname = process.name_()

process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
process.MessageLogger.categories.append('L1GtTrigReport')
process.MessageLogger.categories.append('HLTrigReport')

