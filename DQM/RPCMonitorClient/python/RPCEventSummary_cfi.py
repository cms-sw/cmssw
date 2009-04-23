import FWCore.ParameterSet.Config as cms

rpcEventSummary = cms.EDAnalyzer("RPCEventSummary",
   EventInfoPath = cms.untracked.string('RPC/EventInfo'),
   RPCPrefixDir =  cms.untracked.string('RPC/RecHits'),
   PrescaleFactor = cms.untracked.int32(10),
   MinimunHitsPerRoll = cms.untracked.uint32(2000),
   Tier0 = cms.untracked.bool(False)                              
)
