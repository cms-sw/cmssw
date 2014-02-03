import FWCore.ParameterSet.Config as cms

##
## run the L1 unpacker for GMT, GCT and Castor
##

import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
unpackGctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone(
    inputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess())
)

import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
unpackGtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone(
    DaqGtInputTag = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess())
)

import EventFilter.CastorRawToDigi.CastorRawToDigi_cfi
unpackCastorDigis = EventFilter.CastorRawToDigi.CastorRawToDigi_cfi.castorDigis.clone(
    InputLabel = cms.InputTag( 'rawDataCollector' )
)

##
## run the L1 emulator
##

import L1Trigger.GlobalTrigger.gtDigis_cfi
simGtDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone(
    GmtInputTag                 = cms.InputTag( 'unpackGtDigis' ),
    GctInputTag                 = cms.InputTag( 'unpackGctDigis' ),
    CastorInputTag              = cms.InputTag( 'unpackCastorDigis' ),
    TechnicalTriggersInputTags  = cms.VInputTag()
)

##
## repack FEDs 812 and 813
##

import EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi
l1GtPack = EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi.l1GtPack.clone(
    DaqGtInputTag    = 'simGtDigis',
    MuGmtInputTag    = 'unpackGtDigis'
)
import EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmPack_cfi
l1GtEvmPack = EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmPack_cfi.l1GtEvmPack.clone(
    EvmGtInputTag = 'simGtDigis'
)

##
## combine the new L1 RAW with existing RAW for other FEDs
##

import EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi
rawDataCollector = EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi.rawDataCollector.clone(
    verbose = cms.untracked.int32(0),
    RawCollectionList = cms.VInputTag(
        cms.InputTag('l1GtPack'),
        cms.InputTag('l1GtEvmPack'),
        cms.InputTag('rawDataCollector', processName=cms.InputTag.skipCurrentProcess())
    )
)

SimL1Emulator = cms.Sequence(
      unpackGctDigis
    + unpackGtDigis
    + unpackCastorDigis
    + simGtDigis
    + l1GtPack
    + l1GtEvmPack
    + rawDataCollector
)

