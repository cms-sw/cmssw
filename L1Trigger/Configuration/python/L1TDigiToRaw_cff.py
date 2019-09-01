#
#  L1TDigiToRaw:  Defines
#
#     L1TDigiToRaw = cms.Sequence(...)
#
# which contains all L1 trigger packers needed for the current era.
#
import FWCore.ParameterSet.Config as cms
import sys

# Modify the Raw Data Collection Raw collection List to include upgrade collections where appropriate:
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
stage1L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("caloStage1Raw")) )
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.extend([cms.InputTag("caloLayer1RawFed1354"), cms.InputTag("caloLayer1RawFed1356"), cms.InputTag("caloLayer1RawFed1358")]) )
stage2L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("bmtfStage2Raw")) )
stage2L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("caloStage2Raw")) )
stage2L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("gmtStage2Raw")) )
stage2L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("gtStage2Raw")) )

#
# Legacy Trigger:
#
# legacy L1 packages:
from EventFilter.CSCTFRawToDigi.csctfpacker_cfi import *
from EventFilter.DTTFRawToDigi.dttfpacker_cfi import *
from EventFilter.GctRawToDigi.gctDigiToRaw_cfi import *
from EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi import *
from EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmPack_cfi import *
csctfpacker.lctProducer = "simCscTriggerPrimitiveDigis:MPCSORTED"
csctfpacker.trackProducer = 'simCsctfTrackDigis'
dttfpacker.DTDigi_Source = 'simDtTriggerPrimitiveDigis'
dttfpacker.DTTracks_Source = "simDttfDigis:DTTF"
gctDigiToRaw.rctInputLabel = 'simRctDigis'
gctDigiToRaw.gctInputLabel = 'simGctDigis'
l1GtPack.DaqGtInputTag = 'simGtDigis'
l1GtPack.MuGmtInputTag = 'simGmtDigis'
l1GtEvmPack.EvmGtInputTag = 'simGtDigis'
L1TDigiToRawTask = cms.Task(csctfpacker, dttfpacker, gctDigiToRaw, l1GtPack, l1GtEvmPack)

#
# Stage-1 Trigger
#
# legacy L1 packers, still in use for 2015:
# Initially, the stage-1 was packed via GCT... this is no longer needed.
# (but still needed for RCT digis!)
(stage1L1Trigger & ~stage2L1Trigger).toModify(gctDigiToRaw, gctInputLabel = 'simCaloStage1LegacyFormatDigis')
from EventFilter.L1TRawToDigi.caloStage1Raw_cfi import *
(stage1L1Trigger & ~stage2L1Trigger).toReplaceWith(L1TDigiToRawTask, cms.Task(csctfpacker, dttfpacker, l1GtPack, caloStage1Raw))

#
# Stage-2 Trigger
#
from EventFilter.L1TRawToDigi.caloLayer1Raw_cfi import *
from EventFilter.L1TRawToDigi.caloStage2Raw_cfi import *
from EventFilter.L1TRawToDigi.bmtfStage2Raw_cfi import *
from EventFilter.L1TRawToDigi.omtfStage2Raw_cfi import *
from EventFilter.L1TRawToDigi.gmtStage2Raw_cfi import *
from EventFilter.L1TRawToDigi.gtStage2Raw_cfi import *
# Missing: muon EMTF
(stage2L1Trigger).toReplaceWith(L1TDigiToRawTask, cms.Task(caloLayer1Raw, caloStage2Raw, bmtfStage2Raw, omtfStage2Raw, gmtStage2Raw, gtStage2Raw))

L1TDigiToRaw = cms.Sequence(L1TDigiToRawTask)
