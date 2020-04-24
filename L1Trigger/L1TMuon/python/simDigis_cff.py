import FWCore.ParameterSet.Config as cms
import sys
#
# Legacy L1 Muon modules still running in 2016 trigger:
#

#  - DT TP emulator
from L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi import *
import L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi
simDtTriggerPrimitiveDigis = L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi.dtTriggerPrimitiveDigis.clone()

simDtTriggerPrimitiveDigis.digiTag = 'simMuonDTDigis'
#simDtTriggerPrimitiveDigis.debug = cms.untracked.bool(True)

# - CSC TP emulator
from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
import L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi
simCscTriggerPrimitiveDigis = L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi.cscTriggerPrimitiveDigis.clone()
simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'simMuonCSCDigis', 'MuonCSCComparatorDigi' )
simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'simMuonCSCDigis', 'MuonCSCWireDigi' )

SimL1TMuonCommon = cms.Sequence(simDtTriggerPrimitiveDigis + simCscTriggerPrimitiveDigis)

#
# Legacy Trigger:
#
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
if not (stage2L1Trigger.isChosen()):
#
# - CSC Track Finder emulator
#
    import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
    simCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
    simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag( 'simCscTriggerPrimitiveDigis', 'MPCSORTED' )
    simCsctfTrackDigis.DTproducer = 'simDtTriggerPrimitiveDigis'
    import L1Trigger.CSCTrackFinder.csctfDigis_cfi
    simCsctfDigis = L1Trigger.CSCTrackFinder.csctfDigis_cfi.csctfDigis.clone()
    simCsctfDigis.CSCTrackProducer = 'simCsctfTrackDigis'
#
# - DT Track Finder emulator
# 
    import L1Trigger.DTTrackFinder.dttfDigis_cfi
    simDttfDigis = L1Trigger.DTTrackFinder.dttfDigis_cfi.dttfDigis.clone()
    simDttfDigis.DTDigi_Source  = 'simDtTriggerPrimitiveDigis'
    simDttfDigis.CSCStub_Source = 'simCsctfTrackDigis'
#
# - RPC PAC Trigger emulator
#
    from L1Trigger.RPCTrigger.rpcTriggerDigis_cff import *
    simRpcTriggerDigis = L1Trigger.RPCTrigger.rpcTriggerDigis_cff.rpcTriggerDigis.clone()
    simRpcTriggerDigis.label = 'simMuonRPCDigis'
#
# - Global Muon Trigger emulator
#
    import L1Trigger.GlobalMuonTrigger.gmtDigis_cfi
    simGmtDigis = L1Trigger.GlobalMuonTrigger.gmtDigis_cfi.gmtDigis.clone()
    simGmtDigis.DTCandidates   = cms.InputTag( 'simDttfDigis', 'DT' )
    simGmtDigis.CSCCandidates  = cms.InputTag( 'simCsctfDigis', 'CSC' )
    simGmtDigis.RPCbCandidates = cms.InputTag( 'simRpcTriggerDigis', 'RPCb' )
    simGmtDigis.RPCfCandidates = cms.InputTag( 'simRpcTriggerDigis', 'RPCf' )
#   Note: GMT requires input from calorimeter emulators, namely MipIsoData from GCT
    simGmtDigis.MipIsoData     = 'simRctDigis'
#
#
    SimL1TMuon = cms.Sequence(SimL1TMuonCommon + simCsctfTrackDigis + simCsctfDigis + simDttfDigis + simRpcTriggerDigis + simGmtDigis)

#
# Stage-2 Trigger
#
if stage2L1Trigger.isChosen():
    from L1Trigger.L1TTwinMux.simTwinMuxDigis_cfi import *
    from L1Trigger.L1TMuonBarrel.simBmtfDigis_cfi import *
    from L1Trigger.L1TMuonEndCap.simEmtfDigis_cfi import *
    from L1Trigger.L1TMuonOverlap.simOmtfDigis_cfi import *
    from L1Trigger.L1TMuon.simGmtCaloSumDigis_cfi import *
    from L1Trigger.L1TMuon.simGmtStage2Digis_cfi import *
#
#
    SimL1TMuon = cms.Sequence(SimL1TMuonCommon + simTwinMuxDigis + simBmtfDigis + simEmtfDigis + simOmtfDigis + simGmtCaloSumDigis + simGmtStage2Digis)

    from L1Trigger.ME0Trigger.me0TriggerPseudoDigis_cff import *
    _phase2_SimL1TMuon = SimL1TMuon.copy()
    _phase2_SimL1TMuon += me0TriggerPseudoDigiSequence

    from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
    phase2_muon.toReplaceWith( SimL1TMuon, _phase2_SimL1TMuon )
