import FWCore.ParameterSet.Config as cms

# L1 Emulator sequence for simulation use-case
# Jim Brooke, 24 April 2008

# RCT emulator
import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
simRctDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()

# GCT emulator
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
simGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()

# DT Trigger emulator
from L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi import *
#import L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi
simDtTriggerPrimitiveDigis = L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi.dtTriggerPrimitiveDigis.clone()

# DT Track Finder emulator
import L1Trigger.DTTrackFinder.dttfDigis_cfi
simDttfDigis = L1Trigger.DTTrackFinder.dttfDigis_cfi.dttfDigis.clone()

# CSC Trigger emulator
from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
import L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi
simCscTriggerPrimitiveDigis = L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi.cscTriggerPrimitiveDigis.clone()

# CSC Track Finder emulator
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
simCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
import L1Trigger.CSCTrackFinder.csctfDigis_cfi
simCsctfDigis = L1Trigger.CSCTrackFinder.csctfDigis_cfi.csctfDigis.clone()

# RPC Trigger emulator
from L1Trigger.RPCTrigger.rpcTriggerDigis_cff import *
#import L1Trigger.RPCTrigger.rpcTriggerDigis_cff
simRpcTriggerDigis = L1Trigger.RPCTrigger.rpcTriggerDigis_cff.rpcTriggerDigis.clone()

# Global Muon Trigger emulator
import L1Trigger.GlobalMuonTrigger.gmtDigis_cfi
simGmtDigis = L1Trigger.GlobalMuonTrigger.gmtDigis_cfi.gmtDigis.clone()

# Global Trigger emulator
import L1Trigger.GlobalTrigger.gtDigis_cfi
simGtDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()

# producers for Technical Trigger
#
# BSC Trigger
import L1TriggerOffline.L1Analyzer.bscTrigger_cfi
simBscDigis = L1TriggerOffline.L1Analyzer.bscTrigger_cfi.bscTrigger.clone()

# L1 Trigger sequences
SimL1MuTriggerPrimitives = cms.Sequence(simCscTriggerPrimitiveDigis+simDtTriggerPrimitiveDigis)
SimL1MuTrackFinders = cms.Sequence(simCsctfTrackDigis*simCsctfDigis*simDttfDigis)

SimL1TechnicalTriggers = cms.Sequence(simBscDigis)

SimL1Emulator = cms.Sequence(
    simRctDigis*simGctDigis
    *SimL1MuTriggerPrimitives*SimL1MuTrackFinders*simRpcTriggerDigis*simGmtDigis
    *SimL1TechnicalTriggers
    *simGtDigis)

# correct input tags for MC
simRctDigis.ecalDigisLabel = cms.VInputTag(cms.InputTag("ecalTriggerPrimitiveDigis"))
simRctDigis.hcalDigisLabel = cms.VInputTag(cms.InputTag("hcalTriggerPrimitiveDigis"))
simGctDigis.inputLabel = 'simRctDigis'

simDtTriggerPrimitiveDigis.digiTag = 'simMuonDTDigis'
simDttfDigis.DTDigi_Source = 'simDtTriggerPrimitiveDigis'
simDttfDigis.CSCStub_Source = 'simCsctfTrackDigis'

simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis",
                                                                     "MuonCSCComparatorDigi")
simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis",
                                                               "MuonCSCWireDigi")

simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("simCscTriggerPrimitiveDigis",
                                                                "MPCSORTED")
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
simGtDigis.TechnicalTriggersInputTags = cms.VInputTag(cms.InputTag('simBscDigis'))

