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

# Lookup tables for the CSC TP emulator
from CalibMuon.CSCCalibration.CSCL1TPLookupTableEP_cff import *
# - CSC TP emulator
import L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi
simCscTriggerPrimitiveDigis = L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi.cscTriggerPrimitiveDigis.clone(
    CSCComparatorDigiProducer = 'simMuonCSCDigis:MuonCSCComparatorDigi',
    CSCWireDigiProducer       = 'simMuonCSCDigis:MuonCSCWireDigi'
)
# For Run-3: turn on CCLUT in the MEX/1 chambers
simCscTriggerPrimitiveDigisRun3 = simCscTriggerPrimitiveDigis.clone()
simCscTriggerPrimitiveDigisRun3.commonParam.runCCLUT_OTMB = True

# For Phase-2: turn on CCLUT in the ME1/3 and MEX/2 chambers
simCscTriggerPrimitiveDigisPhase2 = simCscTriggerPrimitiveDigisRun3.clone()
simCscTriggerPrimitiveDigisPhase2.commonParam.runCCLUT_TMB = True

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
from L1Trigger.L1TMuonBarrel.simKBmtfStubs_cfi import *
from L1Trigger.L1TMuonBarrel.simKBmtfDigis_cfi import *
from L1Trigger.L1TMuonEndCap.simEmtfDigis_cfi import *
from L1Trigger.L1TMuonEndCap.simEmtfShowers_cfi import *
from L1Trigger.L1TMuonOverlapPhase1.simOmtfDigis_cfi import *
from L1Trigger.L1TMuon.simGmtCaloSumDigis_cfi import *
from L1Trigger.L1TMuon.simGmtStage2Digis_cfi import *
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
#
#
stage2L1Trigger.toReplaceWith(SimL1TMuonTask, cms.Task(SimL1TMuonCommonTask, simTwinMuxDigis, simBmtfDigis, simKBmtfStubs, simKBmtfDigis, simEmtfDigis, simOmtfDigis, simGmtCaloSumDigis, simGmtStage2Digis))

## hadronic shower trigger for Run-3
from Configuration.Eras.Modifier_run3_common_cff import run3_common
_run3_Shower_SimL1TMuonTask = SimL1TMuonTask.copy()
_run3_Shower_SimL1TMuonTask.add(simEmtfShowers)
_run3_Shower_SimL1TMuonTask.add(simGmtShowerDigis)
(stage2L1Trigger & run3_common).toReplaceWith( SimL1TMuonTask, _run3_Shower_SimL1TMuonTask )

#
# Phase-2 Trigger
#
from L1Trigger.L1TMuonBarrel.simKBmtfStubs_cfi import *
from L1Trigger.L1TMuonBarrel.simKBmtfDigis_cfi import *
from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger
phase2_trigger.toReplaceWith(SimL1TMuonTask, cms.Task(SimL1TMuonCommonTask, simTwinMuxDigis, simBmtfDigis, simKBmtfStubs, simKBmtfDigis, simEmtfDigis, simOmtfDigis, simGmtCaloSumDigis, simGmtStage2Digis, simEmtfShowers, simGmtShowerDigis))

## GEM TPs
from L1Trigger.L1TGEM.simGEMDigis_cff import *
_run3_SimL1TMuonTask = SimL1TMuonTask.copy()
_run3_SimL1TMuonTask.add(simCscTriggerPrimitiveDigisRun3)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
(stage2L1Trigger & run3_GEM).toReplaceWith( SimL1TMuonTask, cms.Task(simMuonGEMPadTask,_run3_SimL1TMuonTask) )

## ME0 TPs
from L1Trigger.L1TGEM.me0TriggerDigis_cff import *
_phase2_SimL1TMuonTask = SimL1TMuonTask.copy()
_phase2_SimL1TMuonTask.add(me0TriggerAllDigiTask)
_phase2_SimL1TMuonTask.add(simCscTriggerPrimitiveDigisPhase2)
# _phase2_GE0_SimL1TMuonTask = SimL1TMuonTask.copyAndExclude([me0TriggerAllDigiTask])

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
(stage2L1Trigger & phase2_muon).toReplaceWith( SimL1TMuonTask, _phase2_SimL1TMuonTask )

# from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
# (stage2L1Trigger & phase2_GE0).toReplaceWith( SimL1TMuonTask, _phase2_GE0_SimL1TMuonTask )
