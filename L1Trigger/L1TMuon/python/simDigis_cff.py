import FWCore.ParameterSet.Config as cms
import sys
#
# Legacy L1 Muon modules still running in 2016 trigger:
#

#  - DT TP emulator
from L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi import *
import L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi
simDtTriggerPrimitiveDigis = L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi.dtTriggerPrimitiveDigis.clone(
    digiTag = 'simMuonDTDigis'
)
#simDtTriggerPrimitiveDigis.debug = cms.untracked.bool(True)

# - CSC TP emulator
from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
import L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi
simCscTriggerPrimitiveDigis = L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi.cscTriggerPrimitiveDigis.clone(
    CSCComparatorDigiProducer = 'simMuonCSCDigis:MuonCSCComparatorDigi',
    CSCWireDigiProducer       = 'simMuonCSCDigis:MuonCSCWireDigi'
)

SimL1TMuonCommonTask = cms.Task(simDtTriggerPrimitiveDigis, simCscTriggerPrimitiveDigis)
SimL1TMuonCommon = cms.Sequence(SimL1TMuonCommonTask)

#
# Legacy Trigger:
#
#
# - CSC Track Finder emulator
#
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
simCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone(
    SectorReceiverInput = 'simCscTriggerPrimitiveDigis:MPCSORTED',
    DTproducer = 'simDtTriggerPrimitiveDigis'
)
import L1Trigger.CSCTrackFinder.csctfDigis_cfi
simCsctfDigis = L1Trigger.CSCTrackFinder.csctfDigis_cfi.csctfDigis.clone(
    CSCTrackProducer = 'simCsctfTrackDigis'
)
#
# - DT Track Finder emulator
#
import L1Trigger.DTTrackFinder.dttfDigis_cfi
simDttfDigis = L1Trigger.DTTrackFinder.dttfDigis_cfi.dttfDigis.clone(
    DTDigi_Source  = 'simDtTriggerPrimitiveDigis',
    CSCStub_Source = 'simCsctfTrackDigis'
)
#
# - RPC PAC Trigger emulator
#
from L1Trigger.RPCTrigger.rpcTriggerDigis_cff import *
simRpcTriggerDigis = L1Trigger.RPCTrigger.rpcTriggerDigis_cff.rpcTriggerDigis.clone(
    label = 'simMuonRPCDigis'
)
#
# - Global Muon Trigger emulator
#
import L1Trigger.GlobalMuonTrigger.gmtDigis_cfi
simGmtDigis = L1Trigger.GlobalMuonTrigger.gmtDigis_cfi.gmtDigis.clone(
    DTCandidates   = 'simDttfDigis:DT',
    CSCCandidates  = 'simCsctfDigis:CSC',
    RPCbCandidates = 'simRpcTriggerDigis:RPCb',
    RPCfCandidates = 'simRpcTriggerDigis:RPCf',
#   Note: GMT requires input from calorimeter emulators, namely MipIsoData from GCT
    MipIsoData     = 'simRctDigis'
)
#
#
SimL1TMuonTask = cms.Task(SimL1TMuonCommonTask, simCsctfTrackDigis, simCsctfDigis, simDttfDigis, simRpcTriggerDigis, simGmtDigis)
SimL1TMuon = cms.Sequence(SimL1TMuonTask)

#
# Stage-2 Trigger
#
from L1Trigger.L1TTwinMux.simTwinMuxDigis_cfi import *
from L1Trigger.L1TMuonBarrel.simBmtfDigis_cfi import *
from L1Trigger.L1TMuonEndCap.simEmtfDigis_cfi import *
from L1Trigger.L1TMuonOverlap.simOmtfDigis_cfi import *
from L1Trigger.L1TMuon.simGmtCaloSumDigis_cfi import *
from L1Trigger.L1TMuon.simGmtStage2Digis_cfi import *
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
#
#
stage2L1Trigger.toReplaceWith(SimL1TMuonTask, cms.Task(SimL1TMuonCommonTask, simTwinMuxDigis, simBmtfDigis, simEmtfDigis, simOmtfDigis, simGmtCaloSumDigis, simGmtStage2Digis))

## GEM TPs
from L1Trigger.L1TGEM.simGEMDigis_cff import *
_run3_SimL1TMuonTask = SimL1TMuonTask.copy()
_run3_SimL1TMuonTask.add(simMuonGEMPadTask)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
(stage2L1Trigger & run3_GEM).toReplaceWith( SimL1TMuonTask, _run3_SimL1TMuonTask )

## ME0 TPs
from L1Trigger.L1TGEM.me0TriggerDigis_cff import *
_phase2_SimL1TMuonTask = SimL1TMuonTask.copy()
_phase2_SimL1TMuonTask.add(me0TriggerAllDigiTask)

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
(stage2L1Trigger & phase2_muon).toReplaceWith( SimL1TMuonTask, _phase2_SimL1TMuonTask )
