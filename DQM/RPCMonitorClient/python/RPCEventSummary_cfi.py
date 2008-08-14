import FWCore.ParameterSet.Config as cms

rpcEventSummary = cms.EDAnalyzer("RPCEventSummary",
   EventInfoPath = cms.untracked.string('RPC/EventInfo'),
   RPCPrefixDir =  cms.untracked.string('RPC/RecHits'),
   PrescaleFactor = cms.untracked.int32(100)
)
