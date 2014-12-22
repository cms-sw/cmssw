import FWCore.ParameterSet.Config as cms

## L1REPACK: redo GCT,GT, using Run-1 or Run-2 input, making Run-2 output
##
## run the L1 unpackers
##

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

from L1Trigger.L1TCalorimeter.L1TCaloStage1_PPFromRaw_cff import *
ecalDigis.InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess())
hcalDigis.InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess())
simHcalTriggerPrimitiveDigis.InputTagFEDRaw = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess())

import L1Trigger.GlobalTrigger.gtDigis_cfi
newGtDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone(
    AlgorithmTriggersUnprescaled= cms.bool(True),
    TechnicalTriggersUnprescaled= cms.bool(True),
    GmtInputTag                 = cms.InputTag( 'unpackGtDigis' ),
    GctInputTag                 = cms.InputTag( 'simCaloStage1LegacyFormatDigis' ),
    CastorInputTag              = cms.InputTag( 'unpackCastorDigis' ),
    TechnicalTriggersInputTags  = cms.VInputTag()
)

##
## pack GCT FEDs
##

import EventFilter.GctRawToDigi.gctDigiToRaw_cfi
packGctDigis = EventFilter.GctRawToDigi.gctDigiToRaw_cfi.gctDigiToRaw.clone(
  gctInputLabel = cms.InputTag( 'simCaloStage1LegacyFormatDigis' )
)

import L1Trigger.L1TCommon.l1tDigiToRaw_cfi
packL1tDigis = L1Trigger.L1TCommon.l1tDigiToRaw_cfi.l1tDigiToRaw.clone(
  InputLabel = cms.InputTag("simCaloStage1FinalDigis"),
  TauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "rlxTaus"),
  IsoTauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "isoTaus"),
  HFBitCountsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFBitCounts"),
  HFRingSumsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFRingSums")
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
        cms.InputTag('packGctDigis'),
        cms.InputTag('packL1tDigis'),
        cms.InputTag('packL1Gt'),
        cms.InputTag('packL1GtEvm'),
        cms.InputTag('rawDataCollector', processName=cms.InputTag.skipCurrentProcess())
    )
)

##
## construct SimL1Emulator sequence
##

SimL1Emulator = cms.Sequence(
      unpackGtDigis      +
      unpackCastorDigis  +
      L1TCaloStage1_PPFromRaw +
      newGtDigis         +
      packGctDigis       +
      packL1tDigis       +
      packL1Gt           +
      packL1GtEvm        +
      rawDataCollector
)
