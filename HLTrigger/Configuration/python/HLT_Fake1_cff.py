# hltGetConfiguration /dev/CMSSW_13_0_0/Fake1 --cff --data --type Fake1

# /dev/CMSSW_13_0_0/Fake1/V5 (CMSSW_13_0_0_pre4)

import FWCore.ParameterSet.Config as cms

fragment = cms.ProcessFragment( "HLT" )


fragment.HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_13_0_0/Fake1/V5')
)

fragment.streams = cms.PSet(  A = cms.vstring( 'InitialPD' ) )
fragment.datasets = cms.PSet(  InitialPD = cms.vstring( 'HLT_Physics_v1',
  'HLT_Random_v1',
  'HLT_ZeroBias_v1' ) )

fragment.CastorDbProducer = cms.ESProducer( "CastorDbProducer",
  appendToDataLabel = cms.string( "" )
)
fragment.hcalDDDRecConstants = cms.ESProducer( "HcalDDDRecConstantsESModule",
  appendToDataLabel = cms.string( "" )
)
fragment.hcalDDDSimConstants = cms.ESProducer( "HcalDDDSimConstantsESModule",
  appendToDataLabel = cms.string( "" )
)

fragment.hltGetConditions = cms.EDAnalyzer( "EventSetupRecordDataGetter",
    verbose = cms.untracked.bool( False ),
    toGet = cms.VPSet( 
    )
)
fragment.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
fragment.hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
fragment.hltTriggerType = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 1 )
)
fragment.hltGtDigis = cms.EDProducer( "L1GlobalTriggerRawToDigi",
    DaqGtInputTag = cms.InputTag( "rawDataCollector" ),
    DaqGtFedId = cms.untracked.int32( 813 ),
    ActiveBoardsMask = cms.uint32( 0xffff ),
    UnpackBxInEvent = cms.int32( 5 ),
    Verbosity = cms.untracked.int32( 0 )
)
fragment.hltCaloStage1Digis = cms.EDProducer( "L1TRawToDigi",
    FedIds = cms.vint32( 1352 ),
    Setup = cms.string( "stage1::CaloSetup" ),
    FWId = cms.uint32( 4294967295 ),
    DmxFWId = cms.uint32( 0 ),
    FWOverride = cms.bool( False ),
    TMTCheck = cms.bool( True ),
    CTP7 = cms.untracked.bool( False ),
    MTF7 = cms.untracked.bool( False ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    lenSlinkHeader = cms.untracked.int32( 8 ),
    lenSlinkTrailer = cms.untracked.int32( 8 ),
    lenAMCHeader = cms.untracked.int32( 8 ),
    lenAMCTrailer = cms.untracked.int32( 0 ),
    lenAMC13Header = cms.untracked.int32( 8 ),
    lenAMC13Trailer = cms.untracked.int32( 8 ),
    debug = cms.untracked.bool( False ),
    MinFeds = cms.uint32( 0 )
)
fragment.hltCaloStage1LegacyFormatDigis = cms.EDProducer( "L1TCaloUpgradeToGCTConverter",
    bxMin = cms.int32( 0 ),
    bxMax = cms.int32( 0 ),
    InputCollection = cms.InputTag( "hltCaloStage1Digis" ),
    InputRlxTauCollection = cms.InputTag( 'hltCaloStage1Digis','rlxTaus' ),
    InputIsoTauCollection = cms.InputTag( 'hltCaloStage1Digis','isoTaus' ),
    InputHFSumsCollection = cms.InputTag( 'hltCaloStage1Digis','HFRingSums' ),
    InputHFCountsCollection = cms.InputTag( 'hltCaloStage1Digis','HFBitCounts' )
)
fragment.hltL1GtObjectMap = cms.EDProducer( "L1GlobalTrigger",
    GmtInputTag = cms.InputTag( "hltGtDigis" ),
    GctInputTag = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    CastorInputTag = cms.InputTag( "castorL1Digis" ),
    TechnicalTriggersInputTags = cms.VInputTag( 'simBscDigis' ),
    ProduceL1GtDaqRecord = cms.bool( False ),
    ProduceL1GtEvmRecord = cms.bool( False ),
    ProduceL1GtObjectMapRecord = cms.bool( True ),
    WritePsbL1GtDaqRecord = cms.bool( False ),
    ReadTechnicalTriggerRecords = cms.bool( True ),
    EmulateBxInEvent = cms.int32( 1 ),
    RecordLength = cms.vint32( 3, 0 ),
    AlternativeNrBxBoardDaq = cms.uint32( 0 ),
    AlternativeNrBxBoardEvm = cms.uint32( 0 ),
    BstLengthBytes = cms.int32( -1 ),
    AlgorithmTriggersUnprescaled = cms.bool( True ),
    AlgorithmTriggersUnmasked = cms.bool( False ),
    TechnicalTriggersUnprescaled = cms.bool( True ),
    TechnicalTriggersUnmasked = cms.bool( False ),
    TechnicalTriggersVetoUnmasked = cms.bool( True )
)
fragment.hltL1extraParticles = cms.EDProducer( "L1ExtraParticlesProd",
    muonSource = cms.InputTag( "hltGtDigis" ),
    etTotalSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    nonIsolatedEmSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','nonIsoEm' ),
    etMissSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    htMissSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    produceMuonParticles = cms.bool( True ),
    forwardJetSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','forJets' ),
    centralJetSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','cenJets' ),
    produceCaloParticles = cms.bool( True ),
    tauJetSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','tauJets' ),
    isoTauJetSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','isoTauJets' ),
    isolatedEmSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','isoEm' ),
    etHadSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    hfRingEtSumsSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    hfRingBitCountsSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    centralBxOnly = cms.bool( True ),
    ignoreHtMiss = cms.bool( False )
)
fragment.hltScalersRawToDigi = cms.EDProducer( "ScalersRawToDigi",
    scalersInputTag = cms.InputTag( "rawDataCollector" )
)
fragment.hltOnlineBeamSpot = cms.EDProducer( "BeamSpotOnlineProducer",
    changeToCMSCoordinates = cms.bool( False ),
    maxZ = cms.double( 40.0 ),
    setSigmaZ = cms.double( 0.0 ),
    beamMode = cms.untracked.uint32( 11 ),
    src = cms.InputTag( "hltScalersRawToDigi" ),
    gtEvmLabel = cms.InputTag( "" ),
    maxRadius = cms.double( 2.0 ),
    useTransientRecord = cms.bool( False )
)
fragment.hltPrePhysics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
fragment.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
fragment.hltRandomEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 3 )
)
fragment.hltPreRandom = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
fragment.hltL1sL1ZeroBias = cms.EDFilter( "HLTLevel1GTSeed",
    saveTags = cms.bool( True ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
fragment.hltPreZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
fragment.hltFEDSelector = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1023 )
)
fragment.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    throw = cms.bool( False ),
    processName = cms.string( "@" ),
    moduleLabelPatternsToMatch = cms.vstring( 'hlt*' ),
    moduleLabelPatternsToSkip = cms.vstring(  )
)
fragment.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
fragment.hltPreHLTAnalyzerEndpath = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
)
fragment.hltL1GtTrigReport = cms.EDAnalyzer( "L1GtTrigReport",
    UseL1GlobalTriggerRecord = cms.bool( False ),
    L1GtRecordInputTag = cms.InputTag( "hltGtDigis" ),
    PrintVerbosity = cms.untracked.int32( 10 ),
    PrintOutput = cms.untracked.int32( 3 )
)
fragment.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','@currentProcess' ),
    reportBy = cms.untracked.string( "job" ),
    resetBy = cms.untracked.string( "never" ),
    serviceBy = cms.untracked.string( "never" ),
    ReferencePath = cms.untracked.string( "HLTriggerFinalPath" ),
    ReferenceRate = cms.untracked.double( 100.0 )
)

fragment.HLTL1UnpackerSequence = cms.Sequence( fragment.hltGtDigis + fragment.hltCaloStage1Digis + fragment.hltCaloStage1LegacyFormatDigis + fragment.hltL1GtObjectMap + fragment.hltL1extraParticles )
fragment.HLTBeamSpot = cms.Sequence( fragment.hltScalersRawToDigi + fragment.hltOnlineBeamSpot )
fragment.HLTBeginSequence = cms.Sequence( fragment.hltTriggerType + fragment.HLTL1UnpackerSequence + fragment.HLTBeamSpot )
fragment.HLTEndSequence = cms.Sequence( fragment.hltBoolEnd )

fragment.HLTriggerFirstPath = cms.Path( fragment.hltGetConditions + fragment.hltGetRaw + fragment.hltBoolFalse )
fragment.HLT_Physics_v1 = cms.Path( fragment.HLTBeginSequence + fragment.hltPrePhysics + fragment.HLTEndSequence )
fragment.HLT_Random_v1 = cms.Path( fragment.hltRandomEventsFilter + fragment.hltGtDigis + fragment.hltPreRandom + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_v1 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sL1ZeroBias + fragment.hltPreZeroBias + fragment.HLTEndSequence )
fragment.HLTriggerFinalPath = cms.Path( fragment.hltGtDigis + fragment.hltScalersRawToDigi + fragment.hltFEDSelector + fragment.hltTriggerSummaryAOD + fragment.hltTriggerSummaryRAW + fragment.hltBoolFalse )
fragment.HLTAnalyzerEndpath = cms.EndPath( fragment.hltGtDigis + fragment.hltPreHLTAnalyzerEndpath + fragment.hltL1GtTrigReport + fragment.hltTrigReport )


fragment.schedule = cms.Schedule( *(fragment.HLTriggerFirstPath, fragment.HLT_Physics_v1, fragment.HLT_Random_v1, fragment.HLT_ZeroBias_v1, fragment.HLTriggerFinalPath, fragment.HLTAnalyzerEndpath, ))


# dummify hltGetConditions in cff's
if 'hltGetConditions' in fragment.__dict__ and 'HLTriggerFirstPath' in fragment.__dict__ :
    fragment.hltDummyConditions = cms.EDFilter( "HLTBool",
        result = cms.bool( True )
    )
    fragment.HLTriggerFirstPath.replace(fragment.hltGetConditions,fragment.hltDummyConditions)

# add specific customizations
from HLTrigger.Configuration.customizeHLTforALL import customizeHLTforAll
fragment = customizeHLTforAll(fragment,"Fake1")

from HLTrigger.Configuration.customizeHLTforCMSSW import customizeHLTforCMSSW
fragment = customizeHLTforCMSSW(fragment,"Fake1")

# Eras-based customisations
from HLTrigger.Configuration.Eras import modifyHLTforEras
modifyHLTforEras(fragment)

