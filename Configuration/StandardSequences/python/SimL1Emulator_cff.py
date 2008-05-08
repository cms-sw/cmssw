import FWCore.ParameterSet.Config as cms

from L1Trigger.Configuration.L1Config_cff import *
import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
simRctDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
simGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
from Geometry.DTGeometry.dtGeometry_cfi import *
import L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi
simDtTriggerPrimitiveDigis = L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi.dtTriggerPrimitiveDigis.clone()
import L1Trigger.DTTrackFinder.dttfDigis_cfi
simDttfDigis = L1Trigger.DTTrackFinder.dttfDigis_cfi.dttfDigis.clone()
from Geometry.CSCGeometry.cscGeometry_cfi import *
from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
import L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi
simCscTriggerPrimitiveDigis = L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi.cscTriggerPrimitiveDigis.clone()
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
simCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
import L1Trigger.CSCTrackFinder.csctfDigis_cfi
simCsctfDigis = L1Trigger.CSCTrackFinder.csctfDigis_cfi.csctfDigis.clone()
from Geometry.RPCGeometry.rpcGeometry_cfi import *
import L1Trigger.RPCTrigger.rpcTriggerDigis_cfi
simRpcTriggerDigis = L1Trigger.RPCTrigger.rpcTriggerDigis_cfi.rpcTriggerDigis.clone()
import L1Trigger.GlobalMuonTrigger.gmtDigis_cfi
simGmtDigis = L1Trigger.GlobalMuonTrigger.gmtDigis_cfi.gmtDigis.clone()
import L1Trigger.GlobalTrigger.gtDigis_cfi
simGtDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()
import EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi
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
simGtDigis.GmtInputTag = 'simGmtDigis'
simGtDigis.GctInputTag = 'simGctDigis'
simGtDigis.TechnicalTriggersInputTag = 'simTechTrigDigis'
simL1GtRecord.L1GtReadoutRecordTag = 'simGtDigis'

