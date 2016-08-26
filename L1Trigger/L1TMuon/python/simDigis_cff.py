import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

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
if not (eras.stage2L1Trigger.isChosen()):
    print "L1TMuon Sequence configured for Legacy trigger (Run1 and Run 2015). "
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
if eras.stage2L1Trigger.isChosen():
    print "L1TMuon Sequence configured for Stage-2 (2016) trigger. "
    from L1Trigger.L1TMuonBarrel.simTwinMuxDigis_cfi import *
    from L1Trigger.L1TMuonBarrel.simBmtfDigis_cfi import *
    from L1Trigger.L1TMuonEndCap.simEmtfDigis_cfi import *
    from L1Trigger.L1TMuonOverlap.simOmtfDigis_cfi import *
    from L1Trigger.L1TMuon.simGmtCaloSumDigis_cfi import *
    from L1Trigger.L1TMuon.simGmtStage2Digis_cfi import *
#
#
    SimL1TMuon = cms.Sequence(SimL1TMuonCommon + simTwinMuxDigis + simBmtfDigis + simEmtfDigis + simOmtfDigis + simGmtCaloSumDigis + simGmtStage2Digis)
