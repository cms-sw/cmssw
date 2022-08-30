#
#  L1TRawToDigi:  Defines
#
#     L1TRawToDigi = cms.Sequence(...)
#
# which contains all packers needed for the current era.
#

import FWCore.ParameterSet.Config as cms
import sys


#
# Legacy Trigger:
#
import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone(producer = 'rawDataCollector')
import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
dttfDigis = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone(DTTF_FED_Source = 'rawDataCollector')
import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
gctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone(inputLabel = 'rawDataCollector')
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone(DaqGtInputTag = 'rawDataCollector')
import EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi
gtEvmDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi.l1GtEvmUnpack.clone(EvmGtInputTag = 'rawDataCollector')
L1TRawToDigi_Legacy = cms.Task(csctfDigis,dttfDigis,gctDigis,gtDigis,gtEvmDigis)
L1TRawToDigiTask = cms.Task(L1TRawToDigi_Legacy)

#
# Stage-1 Trigger
#
from EventFilter.L1TRawToDigi.caloStage1Digis_cfi import caloStage1Digis
# this adds the physical ET to unpacked data
from L1Trigger.L1TCalorimeter.caloStage1LegacyFormatDigis_cfi import caloStage1LegacyFormatDigis
from L1Trigger.L1TCalorimeter.caloStage1FinalDigis_cfi import caloStage1FinalDigis
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
L1TRawToDigi_Stage1 = L1TRawToDigi_Legacy.copyAndExclude([gctDigis, gtDigis, gtEvmDigis])
L1TRawToDigi_Stage1.add(gtDigis,caloStage1Digis,caloStage1FinalDigis,caloStage1LegacyFormatDigis,gctDigis)
(stage1L1Trigger & ~stage2L1Trigger).toReplaceWith(L1TRawToDigiTask, cms.Task(L1TRawToDigi_Stage1))

#
# Stage-2 Trigger:  fow now, unpack Stage 1 and Stage 2 (in case both available)
#
from EventFilter.RPCRawToDigi.rpcTwinMuxRawToDigi_cfi import rpcTwinMuxRawToDigi
from EventFilter.RPCRawToDigi.rpcUnpacker_cfi import rpcunpacker
from EventFilter.RPCRawToDigi.RPCCPPFRawToDigi_cfi import rpcCPPFRawToDigi
from EventFilter.L1TRawToDigi.bmtfDigis_cfi import bmtfDigis
from EventFilter.L1TRawToDigi.omtfStage2Digis_cfi import omtfStage2Digis
from EventFilter.L1TRawToDigi.emtfStage2Digis_cfi import emtfStage2Digis
from EventFilter.L1TRawToDigi.caloLayer1Digis_cfi import caloLayer1Digis
from EventFilter.L1TRawToDigi.caloStage2Digis_cfi import caloStage2Digis
from EventFilter.L1TRawToDigi.gmtStage2Digis_cfi import gmtStage2Digis
from EventFilter.L1TRawToDigi.gtStage2Digis_cfi import gtStage2Digis
from EventFilter.L1TRawToDigi.gtTestcrateStage2Digis_cfi import gtTestcrateStage2Digis
from EventFilter.L1TXRawToDigi.twinMuxStage2Digis_cfi import twinMuxStage2Digis
# we only warn if it is stage-2 era and it is an essential, always present, stage-2 payload:
stage2L1Trigger.toModify(caloStage2Digis, MinFeds = cms.uint32(1))
stage2L1Trigger.toModify(gmtStage2Digis, MinFeds = cms.uint32(1))
stage2L1Trigger.toModify(gtStage2Digis, MinFeds = cms.uint32(1))
L1TRawToDigi_Stage2 = cms.Task(rpcunpacker,rpcTwinMuxRawToDigi, twinMuxStage2Digis, bmtfDigis, omtfStage2Digis, rpcCPPFRawToDigi, emtfStage2Digis, caloLayer1Digis, caloStage2Digis, gmtStage2Digis, gtStage2Digis, gtTestcrateStage2Digis)
stage2L1Trigger.toReplaceWith(L1TRawToDigiTask, cms.Task(L1TRawToDigi_Stage1,L1TRawToDigi_Stage2))

L1TRawToDigi = cms.Sequence(L1TRawToDigiTask)
