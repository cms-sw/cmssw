import FWCore.ParameterSet.Config as cms

rpcEventSummary = cms.EDAnalyzer("RPCEventSummary",
   EventInfoPath = cms.untracked.string('RPC/EventInfo'),
   PrescaleFactor = cms.untracked.int32(10),
   Tier0 = cms.untracked.bool(False),                              
   MinimumRPCEvents = cms.untracked.int32(10000),
   NumberOfEndcapDisks = cms.untracked.int32(3),
   EnableEndcapSummary = cms.untracked.bool(True)                              
)
