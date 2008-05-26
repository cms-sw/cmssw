import FWCore.ParameterSet.Config as cms

# L1 Emulator sequence for simulation use-case
# Jim Brooke, 24 April 2008
# Emulator configuration
from L1Trigger.Configuration.L1Config_cff import *
import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
# RCT emulator
simRctDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
# GCT emulator
simGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
# DT Trigger emulator
from Geometry.DTGeometry.dtGeometry_cfi import *
import L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi
simDtTriggerPrimitiveDigis = L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi.dtTriggerPrimitiveDigis.clone()
import L1Trigger.DTTrackFinder.dttfDigis_cfi
# DT Track Finder emulator
simDttfDigis = L1Trigger.DTTrackFinder.dttfDigis_cfi.dttfDigis.clone()
# CSC Trigger emulator
from Geometry.CSCGeometry.cscGeometry_cfi import *
from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
import L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi
simCscTriggerPrimitiveDigis = L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi.cscTriggerPrimitiveDigis.clone()
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
# CSC Track Finder emulator
simCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
import L1Trigger.CSCTrackFinder.csctfDigis_cfi
simCsctfDigis = L1Trigger.CSCTrackFinder.csctfDigis_cfi.csctfDigis.clone()
# RPC Trigger emulator
from Geometry.RPCGeometry.rpcGeometry_cfi import *
import L1Trigger.RPCTrigger.rpcTriggerDigis_cfi
simRpcTriggerDigis = L1Trigger.RPCTrigger.rpcTriggerDigis_cfi.rpcTriggerDigis.clone()
import L1Trigger.GlobalMuonTrigger.gmtDigis_cfi
# Global Muon Trigger emulator
simGmtDigis = L1Trigger.GlobalMuonTrigger.gmtDigis_cfi.gmtDigis.clone()
import L1Trigger.GlobalTrigger.gtDigis_cfi
# Global Trigger emulator
simGtDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()
import EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi
# Global Trigger AOD producer
simL1GtRecord = EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi.l1GtRecord.clone()
SimL1MuTriggerPrimitives = cms.Sequence(simCscTriggerPrimitiveDigis+simDtTriggerPrimitiveDigis)
SimL1MuTrackFinders = cms.Sequence(simCsctfTrackDigis*simCsctfDigis*simDttfDigis)
SimL1Emulator = cms.Sequence(simRctDigis*simGctDigis*SimL1MuTriggerPrimitives*SimL1MuTrackFinders*simRpcTriggerDigis*simGmtDigis*simGtDigis*simL1GtRecord)
simRctDigis.ecalDigisLabel = 'simEcalTriggerPrimitiveDigis'
simRctDigis.hcalDigisLabel = 'simHcalTriggerPrimitiveDigis'
simGctDigis.inputLabel = 'simRctDigis'
simDtTriggerPrimitiveDigis.digiTag = 'simMuonDTDigis'
simDttfDigis.DTDigi_Source = 'simDtTriggerPrimitiveDigis'
simDttfDigis.CSCStub_Source = 'simCsctfTrackDigis'
simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi")
simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
simCsctfTrackDigis.SectorReceiverInput = cms.InputTag("simCscTriggerPrimitiveDigis","MPCSORTED")
simCsctfTrackDigis.DTproducer = 'simDtTriggerPrimitiveDigis'
simCsctfDigis.CSCTrackProducer = 'simCsctfTrackDigis'
simRpcTriggerDigis.label = 'simMuonRPCDigis'
simGmtDigis.DTCandidates = cms.InputTag("simDttfDigis","DT")
simGmtDigis.CSCCandidates = cms.InputTag("simCsctfDigis","CSC")
simGmtDigis.RPCbCandidates = cms.InputTag("simRpcTriggerDigis","RPCb")
simGmtDigis.RPCfCandidates = cms.InputTag("simRpcTriggerDigis","RPCf")
simGmtDigis.MipIsoData = 'simRctDigis'
simGtDigis.GmtInputTag = 'simGmtDigis'
simGtDigis.GctInputTag = 'simGctDigis'
simGtDigis.TechnicalTriggersInputTag = 'simTechTrigDigis'
simL1GtRecord.L1GtReadoutRecordTag = 'simGtDigis'

