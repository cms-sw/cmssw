# hltGetConfiguration --cff --data /dev/CMSSW_10_1_0/Fake2 --type Fake2

# /dev/CMSSW_10_1_0/Fake2/V8 (CMSSW_10_1_10_HLT1)

import FWCore.ParameterSet.Config as cms

fragment = cms.ProcessFragment( "HLT" )

fragment.HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_10_1_0/Fake2/V8')
)

fragment.streams = cms.PSet(  A = cms.vstring( 'InitialPD' ) )
fragment.datasets = cms.PSet(  InitialPD = cms.vstring( 'HLT_Physics_v1',
  'HLT_Random_v1',
  'HLT_ZeroBias_v1' ) )

fragment.GlobalParametersRcdSource = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "L1TGlobalParametersRcd" ),
    firstValid = cms.vuint32( 1 )
)

fragment.GlobalParameters = cms.ESProducer( "StableParametersTrivialProducer",
  NumberL1JetCounts = cms.uint32( 12 ),
  NumberL1NoIsoEG = cms.uint32( 4 ),
  NumberL1CenJet = cms.uint32( 4 ),
  NumberL1Tau = cms.uint32( 8 ),
  NumberConditionChips = cms.uint32( 1 ),
  NumberL1EGamma = cms.uint32( 12 ),
  TotalBxInEvent = cms.int32( 5 ),
  NumberL1Mu = cms.uint32( 4 ),
  PinsOnConditionChip = cms.uint32( 512 ),
  WordLength = cms.int32( 64 ),
  PinsOnChip = cms.uint32( 512 ),
  OrderOfChip = cms.vint32( 1 ),
  IfMuEtaNumberBits = cms.uint32( 6 ),
  OrderConditionChip = cms.vint32( 1 ),
  appendToDataLabel = cms.string( "" ),
  NumberL1TauJet = cms.uint32( 4 ),
  NumberL1Jet = cms.uint32( 12 ),
  NumberPhysTriggers = cms.uint32( 512 ),
  NumberL1Muon = cms.uint32( 12 ),
  UnitLength = cms.int32( 8 ),
  NumberL1IsoEG = cms.uint32( 4 ),
  NumberTechnicalTriggers = cms.uint32( 64 ),
  NumberL1ForJet = cms.uint32( 4 ),
  IfCaloEtaNumberBits = cms.uint32( 4 ),
  NumberPsbBoards = cms.int32( 7 ),
  NumberChips = cms.uint32( 5 ),
  NumberPhysTriggersExtended = cms.uint32( 64 )
)
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
    toGet = cms.VPSet( 
    ),
    verbose = cms.untracked.bool( False )
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
fragment.hltGtStage2Digis = cms.EDProducer( "L1TRawToDigi",
    lenSlinkTrailer = cms.untracked.int32( 8 ),
    lenAMC13Header = cms.untracked.int32( 8 ),
    CTP7 = cms.untracked.bool( False ),
    lenAMC13Trailer = cms.untracked.int32( 8 ),
    Setup = cms.string( "stage2::GTSetup" ),
    MinFeds = cms.uint32( 0 ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    lenSlinkHeader = cms.untracked.int32( 8 ),
    MTF7 = cms.untracked.bool( False ),
    FWId = cms.uint32( 0 ),
    TMTCheck = cms.bool( True ),
    debug = cms.untracked.bool( False ),
    FedIds = cms.vint32( 1404 ),
    lenAMCHeader = cms.untracked.int32( 8 ),
    lenAMCTrailer = cms.untracked.int32( 0 ),
    FWOverride = cms.bool( False )
)
fragment.hltGtStage2ObjectMap = cms.EDProducer( "L1TGlobalProducer",
    L1DataBxInEvent = cms.int32( 5 ),
    JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    AlgorithmTriggersUnmasked = cms.bool( True ),
    EmulateBxInEvent = cms.int32( 1 ),
    AlgorithmTriggersUnprescaled = cms.bool( True ),
    PrintL1Menu = cms.untracked.bool( False ),
    PrescaleCSVFile = cms.string( "prescale_L1TGlobal.csv" ),
    Verbosity = cms.untracked.int32( 0 ),
    AlgoBlkInputTag = cms.InputTag( "hltGtStage2Digis" ),
    EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    ProduceL1GtDaqRecord = cms.bool( True ),
    PrescaleSet = cms.uint32( 1 ),
    ExtInputTag = cms.InputTag( "hltGtStage2Digis" ),
    EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    TriggerMenuLuminosity = cms.string( "startup" ),
    ProduceL1GtObjectMapRecord = cms.bool( True ),
    AlternativeNrBxBoardDaq = cms.uint32( 0 ),
    GetPrescaleColumnFromData = cms.bool( False ),
    TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    BstLengthBytes = cms.int32( -1 ),
    MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' )
)
fragment.hltScalersRawToDigi = cms.EDProducer( "ScalersRawToDigi",
    scalersInputTag = cms.InputTag( "rawDataCollector" )
)
fragment.hltOnlineBeamSpot = cms.EDProducer( "BeamSpotOnlineProducer",
    maxZ = cms.double( 40.0 ),
    src = cms.InputTag( "hltScalersRawToDigi" ),
    gtEvmLabel = cms.InputTag( "" ),
    changeToCMSCoordinates = cms.bool( False ),
    setSigmaZ = cms.double( 0.0 ),
    maxRadius = cms.double( 2.0 )
)
fragment.hltPrePhysics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
fragment.hltRandomEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 3 )
)
fragment.hltPreRandom = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sZeroBias = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltFEDSelector = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1023, 1024 )
)
fragment.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    moduleLabelPatternsToSkip = cms.vstring(  ),
    processName = cms.string( "@" ),
    moduleLabelPatternsToMatch = cms.vstring( 'hlt*' ),
    throw = cms.bool( False )
)
fragment.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
fragment.hltPreHLTAnalyzerEndpath = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1TGlobalSummary = cms.EDAnalyzer( "L1TGlobalSummary",
    ExtInputTag = cms.InputTag( "hltGtStage2Digis" ),
    MaxBx = cms.int32( 0 ),
    DumpRecord = cms.bool( False ),
    psFileName = cms.string( "prescale_L1TGlobal.csv" ),
    ReadPrescalesFromFile = cms.bool( False ),
    AlgInputTag = cms.InputTag( "hltGtStage2Digis" ),
    MinBx = cms.int32( 0 ),
    psColumn = cms.int32( 0 ),
    DumpTrigResults = cms.bool( False ),
    DumpTrigSummary = cms.bool( True )
)
fragment.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    ReferencePath = cms.untracked.string( "HLTriggerFinalPath" ),
    ReferenceRate = cms.untracked.double( 100.0 ),
    serviceBy = cms.untracked.string( "never" ),
    resetBy = cms.untracked.string( "never" ),
    reportBy = cms.untracked.string( "job" ),
    HLTriggerResults = cms.InputTag( 'TriggerResults','','@currentProcess' )
)

fragment.HLTL1UnpackerSequence = cms.Sequence( fragment.hltGtStage2Digis + fragment.hltGtStage2ObjectMap )
fragment.HLTBeamSpot = cms.Sequence( fragment.hltScalersRawToDigi + fragment.hltOnlineBeamSpot )
fragment.HLTBeginSequence = cms.Sequence( fragment.hltTriggerType + fragment.HLTL1UnpackerSequence + fragment.HLTBeamSpot )
fragment.HLTEndSequence = cms.Sequence( fragment.hltBoolEnd )
fragment.HLTBeginSequenceRandom = cms.Sequence( fragment.hltRandomEventsFilter + fragment.hltGtStage2Digis )

fragment.HLTriggerFirstPath = cms.Path( fragment.hltGetConditions + fragment.hltGetRaw + fragment.hltBoolFalse )
fragment.HLT_Physics_v1 = cms.Path( fragment.HLTBeginSequence + fragment.hltPrePhysics + fragment.HLTEndSequence )
fragment.HLT_Random_v1 = cms.Path( fragment.HLTBeginSequenceRandom + fragment.hltPreRandom + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_v1 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPreZeroBias + fragment.HLTEndSequence )
fragment.HLTriggerFinalPath = cms.Path( fragment.hltGtStage2Digis + fragment.hltScalersRawToDigi + fragment.hltFEDSelector + fragment.hltTriggerSummaryAOD + fragment.hltTriggerSummaryRAW + fragment.hltBoolFalse )
fragment.HLTAnalyzerEndpath = cms.EndPath( fragment.hltGtStage2Digis + fragment.hltPreHLTAnalyzerEndpath + fragment.hltL1TGlobalSummary + fragment.hltTrigReport )


fragment.HLTSchedule = cms.Schedule( *(fragment.HLTriggerFirstPath, fragment.HLT_Physics_v1, fragment.HLT_Random_v1, fragment.HLT_ZeroBias_v1, fragment.HLTriggerFinalPath, fragment.HLTAnalyzerEndpath ))


# dummyfy hltGetConditions in cff's
if 'hltGetConditions' in fragment.__dict__ and 'HLTriggerFirstPath' in fragment.__dict__ :
    fragment.hltDummyConditions = cms.EDFilter( "HLTBool",
        result = cms.bool( True )
    )
    fragment.HLTriggerFirstPath.replace(fragment.hltGetConditions,fragment.hltDummyConditions)

# add specific customizations
from HLTrigger.Configuration.customizeHLTforALL import customizeHLTforAll
fragment = customizeHLTforAll(fragment,"Fake2")

from HLTrigger.Configuration.customizeHLTforCMSSW import customizeHLTforCMSSW
fragment = customizeHLTforCMSSW(fragment,"Fake2")

# Eras-based customisations
from HLTrigger.Configuration.Eras import modifyHLTforEras
modifyHLTforEras(fragment)

