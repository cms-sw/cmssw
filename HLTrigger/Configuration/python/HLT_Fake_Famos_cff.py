# /dev/CMSSW_7_3_0/Fake/V30 (CMSSW_7_3_1_patch2_HLT3)

import FWCore.ParameterSet.Config as cms
from FastSimulation.HighLevelTrigger.HLTSetup_cff import *


HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_7_3_0/Fake/V30')
)

hltGetConditions = cms.EDAnalyzer( "EventSetupRecordDataGetter",
    toGet = cms.VPSet( 
    ),
    verbose = cms.untracked.bool( False )
)
hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
hltPrePhysics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "simGtDigis" ),
    offset = cms.uint32( 0 )
)
hltL1sL1ZeroBias = cms.EDFilter( "HLTLevel1GTSeed",
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
hltPreZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "simGtDigis" ),
    offset = cms.uint32( 0 )
)
hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
)
hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)

HLTriggerFirstPath = cms.Path( hltGetConditions + hltGetRaw + hltBoolFalse )
HLT_Physics_v1 = cms.Path( HLTBeginSequence + hltPrePhysics + cms.SequencePlaceholder( "HLTEndSequence" ) )
HLT_ZeroBias_v1 = cms.Path( HLTBeginSequence + hltL1sL1ZeroBias + hltPreZeroBias + cms.SequencePlaceholder( "HLTEndSequence" ) )
HLTriggerFinalPath = cms.Path( HLTBeginSequence + hltScalersRawToDigi + hltFEDSelector + hltTriggerSummaryAOD + hltTriggerSummaryRAW )


HLTSchedule = cms.Schedule( *(HLTriggerFirstPath, HLT_Physics_v1, HLT_ZeroBias_v1, HLTriggerFinalPath ))




# CMSSW version specific customizations
import os
cmsswVersion = os.environ['CMSSW_VERSION']

# none for now

# dummyfy hltGetConditions in cff's
if 'hltGetConditions' in locals() and 'HLTriggerFirstPath' in locals() :
    hltDummyConditions = cms.EDFilter( "HLTBool",
        result = cms.bool( True )
    )
    HLTriggerFirstPath.replace(hltGetConditions,hltDummyConditions)

