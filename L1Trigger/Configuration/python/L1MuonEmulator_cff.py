import FWCore.ParameterSet.Config as cms

# DT Trigger
from L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi import *
from L1Trigger.DTTrackFinder.dttfDigis_cfi import *
# CSC Trigger
from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi import *
from L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi import *
from L1Trigger.CSCTrackFinder.csctfDigis_cfi import *
# RPC Trigger
from L1Trigger.RPCTrigger.rpcTriggerDigis_cff import *
# Global Muon Trigger
from L1Trigger.GlobalMuonTrigger.gmtDigis_cfi import *
L1MuonTriggerPrimitives = cms.Sequence(cscTriggerPrimitiveDigis*dtTriggerPrimitiveDigis)
L1MuonTrackFinders = cms.Sequence(csctfTrackDigis*csctfDigis*dttfDigis)
L1MuonEmulator = cms.Sequence(L1MuonTriggerPrimitives*L1MuonTrackFinders*rpcTriggerDigis*gmtDigis)

