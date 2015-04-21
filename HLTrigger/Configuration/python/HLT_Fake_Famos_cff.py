# /dev/CMSSW_7_4_0/Fake/V7 (CMSSW_7_4_0)

import FWCore.ParameterSet.Config as cms

fragment = cms.ProcessFragment( "HLT" )

fragment.load( "FastSimulation.HighLevelTrigger.HLTSetup_cff" )

fragment.HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_7_4_0/Fake/V7')
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
fragment.hltPrePhysics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "simGtDigis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sL1ZeroBias = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "l1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "simGtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "simGtDigis" ),
    L1TechTriggerSeeding = cms.bool( False )
)
fragment.hltPreZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "simGtDigis" ),
    offset = cms.uint32( 0 )
)
fragment.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
)
fragment.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)

fragment.HLTriggerFirstPath = cms.Path( fragment.hltGetConditions + fragment.hltGetRaw + fragment.hltBoolFalse )
fragment.HLT_Physics_v1 = cms.Path( fragment.HLTBeginSequence + fragment.hltPrePhysics + cms.SequencePlaceholder( "HLTEndSequence" ) )
fragment.HLT_ZeroBias_v1 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sL1ZeroBias + fragment.hltPreZeroBias + cms.SequencePlaceholder( "HLTEndSequence" ) )
fragment.HLTriggerFinalPath = cms.Path( fragment.HLTBeginSequence + fragment.hltScalersRawToDigi + fragment.hltFEDSelector + fragment.hltTriggerSummaryAOD + fragment.hltTriggerSummaryRAW )


fragment.HLTSchedule = cms.Schedule( *(fragment.HLTriggerFirstPath, fragment.HLT_Physics_v1, fragment.HLT_ZeroBias_v1, fragment.HLTriggerFinalPath ))


# dummyfy hltGetConditions in cff's
if 'hltGetConditions' in fragment.__dict__ and 'HLTriggerFirstPath' in fragment.__dict__ :
    fragment.hltDummyConditions = cms.EDFilter( "HLTBool",
        result = cms.bool( True )
    )
    fragment.HLTriggerFirstPath.replace(fragment.hltGetConditions,fragment.hltDummyConditions)

# add specific customizations
from HLTrigger.Configuration.customizeHLTforALL import customizeHLTforAll
fragment = customizeHLTforAll(fragment)

