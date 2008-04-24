import FWCore.ParameterSet.Config as cms

# L1 Emulator sequence for simulation use-case
# Jim Brooke, 24 April 2008
# Emulator configuration
from L1Trigger.Configuration.L1Config_cff import *
import copy
from L1Trigger.RegionalCaloTrigger.rctDigis_cfi import *
# RCT emulator
simRctDigis = copy.deepcopy(rctDigis)
import copy
from L1Trigger.GlobalCaloTrigger.gctDigis_cfi import *
#replace simRctDigis.ecalDigisLabel = simEcalTriggerPrimitiveDigis
#replace simRctDigis.hcalDigisLabel = simHcalTriggerPrimitiveDigis
# GCT emulator
simGctDigis = copy.deepcopy(gctDigis)
# DT Trigger emulator
from Geometry.DTGeometry.dtGeometry_cfi import *
import copy
from L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi import *
simDtTriggerPrimitiveDigis = copy.deepcopy(dtTriggerPrimitiveDigis)
import copy
from L1Trigger.DTTrackFinder.dttfDigis_cfi import *
#replace simDtTriggerPrimitiveDigis.  // needs fix from Carlo Battilana
# DT Track Finder emulator
simDttfDigis = copy.deepcopy(dttfDigis)
# CSC Trigger emulator
from Geometry.CSCGeometry.cscGeometry_cfi import *
from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
import copy
from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi import *
simCscTriggerPrimitiveDigis = copy.deepcopy(cscTriggerPrimitiveDigis)
import copy
from L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi import *
#replace simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = simMuonCSCDigis:MuonCSCComparatorDigi
#replace simCscTriggerPrimitiveDigis.CSCWireDigiProducer = simMuonCSCDigis:MuonCSCWireDigi
# CSC Track Finder emulator
simCsctfTrackDigis = copy.deepcopy(csctfTrackDigis)
import copy
from L1Trigger.CSCTrackFinder.csctfDigis_cfi import *
simCsctfDigis = copy.deepcopy(csctfDigis)
# RPC Trigger emulator
from Geometry.RPCGeometry.rpcGeometry_cfi import *
import copy
from L1Trigger.RPCTrigger.rpcTriggerDigis_cfi import *
simRpcTriggerDigis = copy.deepcopy(rpcTriggerDigis)
import copy
from L1Trigger.GlobalMuonTrigger.gmtDigis_cfi import *
# Global Muon Trigger emulator
simGmtDigis = copy.deepcopy(gmtDigis)
import copy
from L1Trigger.GlobalTrigger.gtDigis_cfi import *
# Global Trigger emulator
simGtDigis = copy.deepcopy(gtDigis)
import copy
from EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi import *
# Global Trigger AOD producer
simL1GtRecord = copy.deepcopy(l1GtRecord)
SimL1MuTriggerPrimitives = cms.Sequence(simCscTriggerPrimitiveDigis+simDtTriggerPrimitiveDigis)
SimL1MuTrackFinders = cms.Sequence(simCsctfTrackDigis*simCsctfDigis*simDttfDigis)
SimL1Emulator = cms.Sequence(simRctDigis*simGctDigis*SimL1MuTriggerPrimitives*SimL1MuTrackFinders*simGtDigis*simL1GtRecord)
simGctDigis.inputLabel = 'simRctDigis'
simDttfDigis.DTDigi_Source = 'simDtTriggerPrimitiveDigis'
simDttfDigis.CSCStub_Source = 'simCsctfTrackDigis'
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


