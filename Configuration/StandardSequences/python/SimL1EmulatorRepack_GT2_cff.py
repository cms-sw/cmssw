import FWCore.ParameterSet.Config as cms

## L1REPACK: redo GT, using Run-2 input, making Run-2 output

##
## run the L1 unpackers
##

import L1Trigger.L1TCommon.l1tRawToDigi_cfi
unpackGctStage1 = L1Trigger.L1TCommon.l1tRawToDigi_cfi.caloStage1Digis.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess())
)

import L1Trigger.L1TCommon.caloStage1LegacyFormatDigis_cfi
unpackGctDigis = L1Trigger.L1TCommon.caloStage1LegacyFormatDigis_cfi.caloStage1LegacyFormatDigis.clone(
    InputCollection         = cms.InputTag("unpackGctStage1"),
    InputRlxTauCollection   = cms.InputTag("unpackGctStage1:rlxTaus"),
    InputIsoTauCollection   = cms.InputTag("unpackGctStage1:isoTaus"),
    InputHFSumsCollection   = cms.InputTag("unpackGctStage1:HFRingSums"),
    InputHFCountsCollection = cms.InputTag("unpackGctStage1:HFBitCounts")
)

import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
unpackGtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone(
    DaqGtInputTag = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess())
)

import EventFilter.CastorRawToDigi.CastorRawToDigi_cfi
unpackCastorDigis = EventFilter.CastorRawToDigi.CastorRawToDigi_cfi.castorDigis.clone(
    InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess())
)

##
## run the L1 emulator
##

import L1Trigger.GlobalTrigger.gtDigis_cfi
newGtDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone(
    AlgorithmTriggersUnprescaled= cms.bool(True),
    TechnicalTriggersUnprescaled= cms.bool(True),
    GmtInputTag                 = cms.InputTag( 'unpackGtDigis' ),
    GctInputTag                 = cms.InputTag( 'unpackGctDigis' ),
    CastorInputTag              = cms.InputTag( 'unpackCastorDigis' ),
    TechnicalTriggersInputTags  = cms.VInputTag()
)

##
## repack FEDs 812 and 813
##

import EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi
packL1Gt = EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi.l1GtPack.clone(
    DaqGtInputTag    = 'newGtDigis',
    MuGmtInputTag    = 'unpackGtDigis'
)
import EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmPack_cfi
packL1GtEvm = EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmPack_cfi.l1GtEvmPack.clone(
    EvmGtInputTag = 'newGtDigis'
)

##
## combine the new L1 RAW with existing RAW for other FEDs
##

import EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi
rawDataCollector = EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi.rawDataCollector.clone(
    verbose = cms.untracked.int32(0),
    RawCollectionList = cms.VInputTag(
        cms.InputTag('packL1Gt'),
        cms.InputTag('packL1GtEvm'),
        cms.InputTag('rawDataCollector', processName=cms.InputTag.skipCurrentProcess())
    )
)

##
## construct SimL1Emulator sequence
##

SimL1Emulator = cms.Sequence(
      unpackGctStage1    +
      unpackGctDigis     +
      unpackGtDigis      +
      unpackCastorDigis  +
      newGtDigis         +
      packL1Gt           +
      packL1GtEvm        +
      rawDataCollector
)
