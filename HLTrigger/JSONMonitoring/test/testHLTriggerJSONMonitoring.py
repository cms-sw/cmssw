import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.options.numberOfThreads = 1
process.options.wantSummary = True
process.maxEvents.input = 100

process.source = cms.Source('EmptySource')

process.datasets = cms.PSet( 
    TestDatasetX = cms.vstring(
        'HLT_TestPathA_v1',
        'HLT_TestPathB_v1'
    ),
    TestDatasetY = cms.vstring(
        'HLT_TestPathC_v1'
    )
)

process.PrescaleService = cms.Service( "PrescaleService",
    lvl1Labels = cms.vstring(
        "PSColumn0",
        "PSColumn1"
    ),
    lvl1DefaultLabel = cms.string( "PSColumn0" ),
    forceDefault = cms.bool( False ),
    prescaleTable = cms.VPSet(
        cms.PSet(
            pathName = cms.string( "HLT_TestPathA_v1" ),
            prescales = cms.vuint32( 1, 5 )
        ),
        cms.PSet(
            pathName = cms.string( "HLT_TestPathB_v1" ),
            prescales = cms.vuint32( 2, 5 )
        ),
        cms.PSet(
            pathName = cms.string( "HLT_TestPathC_v1" ),
            prescales = cms.vuint32( 1, 5 )
        ),
        cms.PSet(
            pathName = cms.string( "Dataset_TestDatasetY" ),
            prescales = cms.vuint32( 4, 1 )
        )
    )
)

process.hltPSetMap = cms.EDProducer( "ParameterSetBlobProducer" )

process.hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)

process.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)

process.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    throw = cms.bool( False ),
    processName = cms.string( "@" ),
    moduleLabelPatternsToMatch = cms.vstring( 'hlt*' ),
    moduleLabelPatternsToSkip = cms.vstring(  )
)

process.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)

process.hltHLTriggerJSONMonitoring = cms.EDAnalyzer( "HLTriggerJSONMonitoring",
    triggerResults = cms.InputTag( 'TriggerResults::@currentProcess' )
)

process.hltDatasetTestDatasetX = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring(
      'HLT_TestPathA_v1 / 5',
      'HLT_TestPathB_v1 / 3'
    )
)

process.hltPreDatasetTestDatasetX = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "" )
)

process.hltDatasetTestDatasetY = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring(
      'HLT_TestPathC_v1 / 10'
    )
)

process.hltPreDatasetTestDatasetY = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "" )
)

process.hltPreTestPathA = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "" )
)

process.hltPreTestPathB = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "" )
)

process.hltPreTestPathC = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "" )
)

process.HLTDatasetPathBeginSequence = cms.Sequence( )

process.HLTBeginSequence = cms.Sequence( )

process.HLTEndSequence = cms.Sequence( process.hltBoolEnd )

process.HLTriggerFirstPath = cms.Path( process.hltPSetMap + process.hltBoolFalse )

process.HLT_TestPathA_v1 = cms.Path( process.HLTBeginSequence + process.hltPreTestPathA + process.HLTEndSequence )

process.HLT_TestPathB_v1 = cms.Path( process.HLTBeginSequence + process.hltPreTestPathB + process.HLTEndSequence )

process.HLT_TestPathC_v1 = cms.Path( process.HLTBeginSequence + process.hltPreTestPathC + process.HLTEndSequence )

process.HLTriggerFinalPath = cms.Path( process.hltTriggerSummaryAOD + process.hltTriggerSummaryRAW + process.hltBoolFalse )

process.RatesMonitoring = cms.EndPath( process.hltHLTriggerJSONMonitoring )

process.Dataset_TestDatasetX = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetTestDatasetX + process.hltPreDatasetTestDatasetX )

process.Dataset_TestDatasetY = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetTestDatasetY + process.hltPreDatasetTestDatasetY )

process.schedule = cms.Schedule(
  process.HLTriggerFirstPath,
  process.HLT_TestPathA_v1,
  process.HLT_TestPathB_v1,
  process.HLT_TestPathC_v1,
  process.HLTriggerFinalPath,
  process.RatesMonitoring,
  process.Dataset_TestDatasetX,
  process.Dataset_TestDatasetY
)
