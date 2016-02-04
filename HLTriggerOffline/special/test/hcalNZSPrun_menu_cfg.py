import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTdr")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20000) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_2_1_9/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/3C7F5E64-B885-DD11-BBFD-000423D94A04.root',
'/store/relval/CMSSW_2_1_9/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/4E6892BC-B885-DD11-A84B-001617E30CD4.root',
'/store/relval/CMSSW_2_1_9/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/CE5C7A96-B985-DD11-97B9-000423D99F1E.root',
'/store/relval/CMSSW_2_1_9/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/D8D5F240-B985-DD11-8C60-001617C3B73A.root',
'/store/relval/CMSSW_2_1_9/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0000/E01A83C4-B785-DD11-89BB-001617C3B70E.root',
'/store/relval/CMSSW_2_1_9/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/9C42CEDD-0487-DD11-B2A5-000423D99AAE.root'
)
)


### HLT for NZSP HCAL

process.HLTConfigVersion = cms.PSet(
  tableName = cms.string('/user/safronov/219/HcaPhiSym/V5')
)

process.block_AlCa_HcalPhiSym_EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring( 'keep triggerTriggerEvent_*_*_*',
                                            'keep *_hcalFED_*_*',
                                            'keep *_hltGctDigis_*_*',
                                            'keep *_hltGtDigis_*_*',
                                            'keep *_hltL1extraParticles_*_*',
                                            'keep *_hltL1GtObjectMap_*_*'
                                            )
    )

process.l1CaloGeomRecordSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1CaloGeometryRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.L1MuTriggerScalesRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1MuTriggerScalesRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.L1GtTriggerMaskVetoTechTrigRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1GtTriggerMaskVetoTechTrigRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.L1GtTriggerMaskVetoAlgoTrigRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1GtTriggerMaskVetoAlgoTrigRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.L1GtTriggerMaskTechTrigRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1GtTriggerMaskTechTrigRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.L1GtTriggerMaskAlgoTrigRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1GtTriggerMaskAlgoTrigRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.L1GtStableParametersRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1GtStableParametersRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.L1GtPrescaleFactorsTechTrigRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1GtPrescaleFactorsTechTrigRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.L1GtPrescaleFactorsAlgoTrigRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1GtPrescaleFactorsAlgoTrigRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.L1GtParametersRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1GtParametersRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.L1GtBoardMapsRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1GtBoardMapsRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.L1MuTriggerPtScaleRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1MuTriggerPtScaleRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)

process.L1MuTriggerPtScale = cms.ESProducer( "L1MuTriggerPtScaleProducer",
  nbitPackingPt = cms.int32( 5 ),
  signedPackingPt = cms.bool( False ),
  nbinsPt = cms.int32( 32 ),
  scalePt = cms.vdouble( -1.0, 0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 140.0, 1000000.0 )
)
process.L1MuTriggerScales = cms.ESProducer( "L1MuTriggerScalesProducer",
  nbitPackingDTEta = cms.int32( 6 ),
  signedPackingDTEta = cms.bool( True ),
  nbinsDTEta = cms.int32( 64 ),
  minDTEta = cms.double( -1.2 ),
  maxDTEta = cms.double( 1.2 ),
  offsetDTEta = cms.int32( 32 ),
  nbitPackingCSCEta = cms.int32( 6 ),
  nbinsCSCEta = cms.int32( 32 ),
  minCSCEta = cms.double( 0.9 ),
  maxCSCEta = cms.double( 2.5 ),
  nbitPackingBrlRPCEta = cms.int32( 6 ),
  signedPackingBrlRPCEta = cms.bool( True ),
  nbinsBrlRPCEta = cms.int32( 33 ),
  offsetBrlRPCEta = cms.int32( 16 ),
  nbitPackingFwdRPCEta = cms.int32( 6 ),
  signedPackingFwdRPCEta = cms.bool( True ),
  nbinsFwdRPCEta = cms.int32( 33 ),
  offsetFwdRPCEta = cms.int32( 16 ),
  nbitPackingGMTEta = cms.int32( 6 ),
  nbinsGMTEta = cms.int32( 31 ),
  nbitPackingPhi = cms.int32( 8 ),
  signedPackingPhi = cms.bool( False ),
  nbinsPhi = cms.int32( 144 ),
  minPhi = cms.double( 0.0 ),
  maxPhi = cms.double( 6.2831853 ),
  scaleRPCEta = cms.vdouble( -2.1, -1.97, -1.85, -1.73, -1.61, -1.48, -1.36, -1.24, -1.14, -1.04, -0.93, -0.83, -0.72, -0.58, -0.44, -0.27, -0.07, 0.07, 0.27, 0.44, 0.58, 0.72, 0.83, 0.93, 1.04, 1.14, 1.24, 1.36, 1.48, 1.61, 1.73, 1.85, 1.97, 2.1 ),
  scaleGMTEta = cms.vdouble( 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4 )
)
process.l1CaloGeometry = cms.ESProducer( "L1CaloGeometryProd",
  numberGctEmJetPhiBins = cms.uint32( 18 ),
  gctEmJetPhiBinOffset = cms.double( -0.5 ),
  numberGctEtSumPhiBins = cms.uint32( 72 ),
  gctEtSumPhiBinOffset = cms.double( 0.0 ),
  numberGctCentralEtaBinsPerHalf = cms.uint32( 7 ),
  numberGctForwardEtaBinsPerHalf = cms.uint32( 4 ),
  etaSignBitOffset = cms.uint32( 8 ),
  gctEtaBinBoundaries = cms.vdouble( 0.0, 0.348, 0.695, 1.044, 1.392, 1.74, 2.172, 3.0, 3.5, 4.0, 4.5, 5.0 )
)
process.l1GtTriggerMenuXml = cms.ESProducer( "L1GtTriggerMenuXmlProducer",
  TriggerMenuLuminosity = cms.string( "lumi1030" ),
  DefXmlFile = cms.string( "L1Menu2008_2E30.xml" ),
  VmeXmlFile = cms.string( "" )
)
process.l1GtTriggerMaskVetoTechTrig = cms.ESProducer( "L1GtTriggerMaskVetoTechTrigTrivialProducer",
  TriggerMask = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
)
process.l1GtTriggerMaskVetoAlgoTrig = cms.ESProducer( "L1GtTriggerMaskVetoAlgoTrigTrivialProducer",
  TriggerMask = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
)
process.l1GtTriggerMaskTechTrig = cms.ESProducer( "L1GtTriggerMaskTechTrigTrivialProducer",
  TriggerMask = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
)
process.l1GtTriggerMaskAlgoTrig = cms.ESProducer( "L1GtTriggerMaskAlgoTrigTrivialProducer",
  TriggerMask = cms.vuint32( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
)
process.l1GtStableParameters = cms.ESProducer( "L1GtStableParametersTrivialProducer",
  NumberPhysTriggers = cms.uint32( 128 ),
  NumberPhysTriggersExtended = cms.uint32( 64 ),
  NumberTechnicalTriggers = cms.uint32( 64 ),
  NumberL1Mu = cms.uint32( 4 ),
  NumberL1NoIsoEG = cms.uint32( 4 ),
  NumberL1IsoEG = cms.uint32( 4 ),
  NumberL1CenJet = cms.uint32( 4 ),
  NumberL1ForJet = cms.uint32( 4 ),
  NumberL1TauJet = cms.uint32( 4 ),
  NumberL1JetCounts = cms.uint32( 12 ),
  NumberConditionChips = cms.uint32( 2 ),
  PinsOnConditionChip = cms.uint32( 96 ),
  NumberPsbBoards = cms.int32( 7 ),
  IfCaloEtaNumberBits = cms.uint32( 4 ),
  IfMuEtaNumberBits = cms.uint32( 6 ),
  WordLength = cms.int32( 64 ),
  UnitLength = cms.int32( 8 ),
  OrderConditionChip = cms.vint32( 2, 1 )
)
process.l1GtPrescaleFactorsTechTrig = cms.ESProducer( "L1GtPrescaleFactorsTechTrigTrivialProducer",
  PrescaleFactorsSet = cms.VPSet( 
    cms.PSet(  PrescaleFactors = cms.vint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )    )
  )
)
process.l1GtPrescaleFactorsAlgoTrig = cms.ESProducer( "L1GtPrescaleFactorsAlgoTrigTrivialProducer",
  PrescaleFactorsSet = cms.VPSet( 
    cms.PSet(  PrescaleFactors = cms.vint32( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )    )
  )
)
process.l1GtParameters = cms.ESProducer( "L1GtParametersTrivialProducer",
  TotalBxInEvent = cms.int32( 1 ),
  DaqActiveBoards = cms.uint32( 0xffff ),
  EvmActiveBoards = cms.uint32( 0xffff ),
  BstLengthBytes = cms.uint32( 30 )
)
process.l1GtBoardMaps = cms.ESProducer( "L1GtBoardMapsTrivialProducer",
  BoardList = cms.vstring( 'GTFE',
    'FDL',
    'PSB',
    'PSB',
    'PSB',
    'PSB',
    'PSB',
    'PSB',
    'PSB',
    'GMT',
    'TCS',
    'TIM' ),
  BoardIndex = cms.vint32( 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0 ),
  BoardPositionDaqRecord = cms.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -1 ),
  BoardPositionEvmRecord = cms.vint32( 1, 3, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1 ),
  ActiveBoardsDaqRecord = cms.vint32( -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1 ),
  ActiveBoardsEvmRecord = cms.vint32( -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1 ),
  BoardSlotMap = cms.vint32( 17, 10, 9, 13, 14, 15, 19, 20, 21, 18, 7, 16 ),
  BoardHexNameMap = cms.vint32( 0, 253, 187, 187, 187, 187, 187, 187, 187, 221, 204, 173 ),
  CableList = cms.vstring( 'TechTr',
    'Free',
    'Free',
    'Free',
    'IsoEGQ',
    'NoIsoEGQ',
    'CenJetQ',
    'ForJetQ',
    'TauJetQ',
    'ESumsQ',
    'JetCountsQ',
    'Free',
    'Free',
    'Free',
    'Free',
    'Free',
    'MQB1',
    'MQB2',
    'MQF3',
    'MQF4',
    'MQB5',
    'MQB6',
    'MQF7',
    'MQF8',
    'MQB9',
    'MQB10',
    'MQF11',
    'MQF12' ),
  CableToPsbMap = cms.vint32( 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6 )
)
process.l1CaloScales = cms.ESProducer( "L1ScalesTrivialProducer",
  L1CaloEmEtScaleLSB = cms.double( 0.5 ),
  L1CaloRegionEtScaleLSB = cms.double( 0.5 ),
  L1CaloEmThresholds = cms.vdouble( 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0 ),
  L1CaloJetThresholds = cms.vdouble( 0.0, 10.0, 12.0, 14.0, 15.0, 18.0, 20.0, 22.0, 24.0, 25.0, 28.0, 30.0, 32.0, 35.0, 37.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 100.0, 110.0, 120.0, 125.0, 130.0, 140.0, 150.0, 160.0, 170.0, 175.0, 180.0, 190.0, 200.0, 215.0, 225.0, 235.0, 250.0, 275.0, 300.0, 325.0, 350.0, 375.0, 400.0, 425.0, 450.0, 475.0, 500.0, 525.0, 550.0, 575.0, 600.0, 625.0, 650.0, 675.0, 700.0, 725.0, 750.0, 775.0 )
)

process.emrcdsrc = cms.ESSource("EmptyESSource",
   recordName = cms.string('L1EmEtScaleRcd'),
   iovIsRunNotTime = cms.bool(True),
   firstValid = cms.vuint32(1)
)
 
process.jetrcdsrc = cms.ESSource("EmptyESSource",
   recordName = cms.string('L1JetEtScaleRcd'),
   iovIsRunNotTime = cms.bool(True),
   firstValid = cms.vuint32(1)
)

process.L1GctConfigProducers = cms.ESProducer( "L1GctConfigProducers",
  JetFinderCentralJetSeed = cms.uint32( 1 ),
  JetFinderForwardJetSeed = cms.uint32( 1 ),
  L1CaloHtScaleLsbInGeV = cms.double( 1.0 ),
  L1CaloJetZeroSuppressionThresholdInGeV = cms.double( 5.0 ),
  CalibrationStyle = cms.string( "ORCAStyle" ),
  PowerSeriesCoefficients = cms.PSet( 
    nonTauJetCalib0 = cms.vdouble(  ),
    nonTauJetCalib1 = cms.vdouble(  ),
    nonTauJetCalib2 = cms.vdouble(  ),
    nonTauJetCalib3 = cms.vdouble(  ),
    nonTauJetCalib4 = cms.vdouble(  ),
    nonTauJetCalib5 = cms.vdouble(  ),
    nonTauJetCalib6 = cms.vdouble(  ),
    nonTauJetCalib7 = cms.vdouble(  ),
    nonTauJetCalib8 = cms.vdouble(  ),
    nonTauJetCalib9 = cms.vdouble(  ),
    nonTauJetCalib10 = cms.vdouble(  ),
    tauJetCalib0 = cms.vdouble(  ),
    tauJetCalib1 = cms.vdouble(  ),
    tauJetCalib2 = cms.vdouble(  ),
    tauJetCalib3 = cms.vdouble(  ),
    tauJetCalib4 = cms.vdouble(  ),
    tauJetCalib5 = cms.vdouble(  ),
    tauJetCalib6 = cms.vdouble(  )
  ),
  OrcaStyleCoefficients = cms.PSet( 
    nonTauJetCalib0 = cms.vdouble( 47.4, -20.7, 0.7922, 9.53E-5 ),
    nonTauJetCalib1 = cms.vdouble( 49.4, -22.5, 0.7867, 9.6E-5 ),
    nonTauJetCalib2 = cms.vdouble( 47.1, -22.2, 0.7645, 1.209E-4 ),
    nonTauJetCalib3 = cms.vdouble( 49.3, -22.9, 0.7331, 1.221E-4 ),
    nonTauJetCalib4 = cms.vdouble( 48.2, -24.5, 0.7706, 1.28E-4 ),
    nonTauJetCalib5 = cms.vdouble( 42.0, -23.9, 0.7945, 1.458E-4 ),
    nonTauJetCalib6 = cms.vdouble( 33.8, -22.1, 0.8202, 1.403E-4 ),
    nonTauJetCalib7 = cms.vdouble( 17.1, -6.6, 0.6958, 6.88E-5 ),
    nonTauJetCalib8 = cms.vdouble( 13.1, -4.5, 0.7071, 7.26E-5 ),
    nonTauJetCalib9 = cms.vdouble( 12.4, -3.8, 0.6558, 4.89E-4 ),
    nonTauJetCalib10 = cms.vdouble( 9.3, 1.3, 0.2719, 0.003418 ),
    tauJetCalib0 = cms.vdouble( 47.4, -20.7, 0.7922, 9.53E-5 ),
    tauJetCalib1 = cms.vdouble( 49.4, -22.5, 0.7867, 9.6E-5 ),
    tauJetCalib2 = cms.vdouble( 47.1, -22.2, 0.7645, 1.209E-4 ),
    tauJetCalib3 = cms.vdouble( 49.3, -22.9, 0.7331, 1.221E-4 ),
    tauJetCalib4 = cms.vdouble( 48.2, -24.5, 0.7706, 1.28E-4 ),
    tauJetCalib5 = cms.vdouble( 42.0, -23.9, 0.7945, 1.458E-4 ),
    tauJetCalib6 = cms.vdouble( 33.8, -22.1, 0.8202, 1.403E-4 )
  ),
  PiecewiseCubicCoefficients = cms.PSet( 
    nonTauJetCalib0 = cms.vdouble( 500.0, 100.0, 17.7409, 0.351901, -7.01462E-4, 5.77204E-7, 5.0, 0.720604, 1.25179, -0.0150777, 7.13711E-5 ),
    nonTauJetCalib1 = cms.vdouble( 500.0, 100.0, 20.0549, 0.321867, -6.4901E-4, 5.50042E-7, 5.0, 1.30465, 1.2774, -0.0159193, 7.64496E-5 ),
    nonTauJetCalib2 = cms.vdouble( 500.0, 100.0, 24.3454, 0.257989, -4.50184E-4, 3.09951E-7, 5.0, 2.1034, 1.32441, -0.0173659, 8.50669E-5 ),
    nonTauJetCalib3 = cms.vdouble( 500.0, 100.0, 27.7822, 0.155986, -2.66441E-4, 6.69814E-8, 5.0, 2.64613, 1.30745, -0.0180964, 8.83567E-5 ),
    nonTauJetCalib4 = cms.vdouble( 500.0, 100.0, 26.6384, 0.0567369, -4.16292E-4, 2.60929E-7, 5.0, 2.63299, 1.16558, -0.0170351, 7.95703E-5 ),
    nonTauJetCalib5 = cms.vdouble( 500.0, 100.0, 29.5396, 0.001137, -1.45232E-4, 6.91445E-8, 5.0, 4.16752, 1.08477, -0.016134, 7.69652E-5 ),
    nonTauJetCalib6 = cms.vdouble( 500.0, 100.0, 30.1405, -0.14281, 5.55849E-4, -7.52446E-7, 5.0, 4.79283, 0.672125, -0.00879174, 3.65776E-5 ),
    nonTauJetCalib7 = cms.vdouble( 300.0, 80.0, 30.2715, -0.539688, 0.00499898, -1.2204E-5, 5.0, 1.97284, 0.0610729, 0.00671548, -7.22583E-5 ),
    nonTauJetCalib8 = cms.vdouble( 250.0, 150.0, 1.38861, 0.0362661, 0.0, 0.0, 5.0, 1.87993, 0.0329907, 0.0, 0.0 ),
    nonTauJetCalib9 = cms.vdouble( 200.0, 80.0, 35.0095, -0.669677, 0.00208498, -1.50554E-6, 5.0, 3.16074, -0.114404, 0.0, 0.0 ),
    nonTauJetCalib10 = cms.vdouble( 150.0, 80.0, 1.70475, -0.142171, 0.00104963, -1.62214E-5, 5.0, 1.70475, -0.142171, 0.00104963, -1.62214E-5 ),
    tauJetCalib0 = cms.vdouble( 500.0, 100.0, 17.7409, 0.351901, -7.01462E-4, 5.77204E-7, 5.0, 0.720604, 1.25179, -0.0150777, 7.13711E-5 ),
    tauJetCalib1 = cms.vdouble( 500.0, 100.0, 20.0549, 0.321867, -6.4901E-4, 5.50042E-7, 5.0, 1.30465, 1.2774, -0.0159193, 7.64496E-5 ),
    tauJetCalib2 = cms.vdouble( 500.0, 100.0, 24.3454, 0.257989, -4.50184E-4, 3.09951E-7, 5.0, 2.1034, 1.32441, -0.0173659, 8.50669E-5 ),
    tauJetCalib3 = cms.vdouble( 500.0, 100.0, 27.7822, 0.155986, -2.66441E-4, 6.69814E-8, 5.0, 2.64613, 1.30745, -0.0180964, 8.83567E-5 ),
    tauJetCalib4 = cms.vdouble( 500.0, 100.0, 26.6384, 0.0567369, -4.16292E-4, 2.60929E-7, 5.0, 2.63299, 1.16558, -0.0170351, 7.95703E-5 ),
    tauJetCalib5 = cms.vdouble( 500.0, 100.0, 29.5396, 0.001137, -1.45232E-4, 6.91445E-8, 5.0, 4.16752, 1.08477, -0.016134, 7.69652E-5 ),
    tauJetCalib6 = cms.vdouble( 500.0, 100.0, 30.1405, -0.14281, 5.55849E-4, -7.52446E-7, 5.0, 4.79283, 0.672125, -0.00879174, 3.65776E-5 )
  ),
  jetCounterSetup = cms.PSet( 
    jetCountersNegativeWheel = cms.VPSet( 
      cms.PSet(  cutDescriptionList = cms.vstring( 'JC_minRank_1' )      ),
      cms.PSet(  cutDescriptionList = cms.vstring( 'JC_minRank_1',
  'JC_centralEta_6' )      ),
      cms.PSet(  cutDescriptionList = cms.vstring( 'JC_minRank_11' )      ),
      cms.PSet(  cutDescriptionList = cms.vstring( 'JC_minRank_11',
  'JC_centralEta_6' )      ),
      cms.PSet(  cutDescriptionList = cms.vstring( 'JC_minRank_19' )      )
    ),
    jetCountersPositiveWheel = cms.VPSet( 
      cms.PSet(  cutDescriptionList = cms.vstring( 'JC_minRank_1' )      ),
      cms.PSet(  cutDescriptionList = cms.vstring( 'JC_minRank_1',
  'JC_centralEta_6' )      ),
      cms.PSet(  cutDescriptionList = cms.vstring( 'JC_minRank_11' )      ),
      cms.PSet(  cutDescriptionList = cms.vstring( 'JC_minRank_11',
  'JC_centralEta_6' )      ),
      cms.PSet(  cutDescriptionList = cms.vstring( 'JC_minRank_19' )      )
    )
  ),
  ConvertEtValuesToEnergy = cms.bool( False )
)

process.hltTriggerType = cms.EDFilter( "TriggerTypeFilter",
    InputLabel = cms.string( "rawDataCollector" ),
    TriggerFedId = cms.int32( 812 ),
    SelectedTriggerType = cms.int32( 1 )
)
process.hltGtDigis = cms.EDProducer( "L1GlobalTriggerRawToDigi",
    DaqGtInputTag = cms.InputTag( "rawDataCollector" ),
    DaqGtFedId = cms.untracked.int32( 813 ),
    ActiveBoardsMask = cms.uint32( 0x101 ),
    UnpackBxInEvent = cms.int32( 1 )
)
process.hltGctDigis = cms.EDProducer( "GctRawToDigi",
    inputLabel = cms.InputTag( "rawDataCollector" ),
    gctFedId = cms.int32( 745 ),
    hltMode = cms.bool( False ),
    grenCompatibilityMode = cms.bool( False )
)
process.hltL1GtObjectMap = cms.EDProducer( "L1GlobalTrigger",
    GmtInputTag = cms.InputTag( "hltGtDigis" ),
    GctInputTag = cms.InputTag( "hltGctDigis" ),
    CastorInputTag = cms.InputTag( "castorL1Digis" ),
    TechnicalTriggersInputTag = cms.InputTag( "techTrigDigis" ),
    ProduceL1GtDaqRecord = cms.bool( False ),
    ProduceL1GtEvmRecord = cms.bool( False ),
    ProduceL1GtObjectMapRecord = cms.bool( True ),
    WritePsbL1GtDaqRecord = cms.bool( False ),
    ReadTechnicalTriggerRecords = cms.bool( True ),
    EmulateBxInEvent = cms.int32( 1 ),
    BstLengthBytes = cms.int32( -1 )
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
    centralBxOnly = cms.bool( True ),
    htMissSource = cms.InputTag("hltGctDigis"),
    hfRingEtSumsSource = cms.InputTag("hltGctDigis"),
    hfRingBitCountsSource = cms.InputTag("hltGctDigis"),
    ignoreHtMiss = cms.bool(False)                                          
)
process.hltOfflineBeamSpot = cms.EDProducer( "BeamSpotProducer" )
process.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
process.hltPreFirstPath = cms.EDFilter( "HLTPrescaler" )
process.hltBoolFirstPath = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
process.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
process.hltL1sHcalPhiSym = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG2 OR L1_DoubleEG1" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
process.hltHcalPhiSymPresc = cms.EDFilter( "HLTPrescaler" )
process.hcalFED = cms.EDProducer( "SubdetFEDSelector",
    getECAL = cms.bool( False ),
    getSiStrip = cms.bool( False ),
    getSiPixel = cms.bool( False ),
    getHCAL = cms.bool( True ),
    getMuon = cms.bool( False ),
    getTrigger = cms.bool( True ),
    rawInputLabel = cms.InputTag( "rawDataCollector" )
)
process.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
)
process.hltPreTriggerSummaryRAW = cms.EDFilter( "HLTPrescaler" )
process.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
process.hltBoolFinalPath = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)

process.HLTBeginSequence = cms.Sequence( process.hltTriggerType + process.hltGtDigis + process.hltGctDigis + process.hltL1GtObjectMap + process.hltL1extraParticles + process.hltOfflineBeamSpot )
process.HLTEndSequence = cms.Sequence( process.hltBoolEnd )

process.HLTriggerFirstPath = cms.Path( process.HLTBeginSequence + process.hltGetRaw + process.hltPreFirstPath + process.hltBoolFirstPath + process.HLTEndSequence )
process.AlCa_HcalPhiSym = cms.Path( process.HLTBeginSequence + process.hltL1sHcalPhiSym + process.hltHcalPhiSymPresc + process.hcalFED + process.HLTEndSequence )
process.HLTriggerFinalPath = cms.Path( process.hltTriggerSummaryAOD + process.hltPreTriggerSummaryRAW + process.hltTriggerSummaryRAW + process.hltBoolFinalPath )


####################

process.load("HLTrigger.Configuration.HLTDefaultOutputWithFEDs_cff")
process.block_hltDefaultOutputWithFEDs.outputCommands.extend(process.block_AlCa_HcalPhiSym_EventContent.outputCommands)
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.block_hltDefaultOutputWithFEDs.outputCommands,
    fileName = cms.untracked.string('AlCaHcalPhiSymStream.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('AlCa_HcalPhiSym')
    )
)

process.out = cms.EndPath(process.output)

