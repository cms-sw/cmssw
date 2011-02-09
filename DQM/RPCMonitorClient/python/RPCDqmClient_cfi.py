import FWCore.ParameterSet.Config as cms

rpcdqmclient = cms.EDAnalyzer("RPCDqmClient",                               
   RPCDqmClientList = cms.untracked.vstring("RPCMultiplicityTest", "RPCDeadChannelTest", "RPCClusterSizeTest", "RPCOccupancyTest","RPCNoisyStripTest"),
   DiagnosticPrescale = cms.untracked.int32(1),
   MinimumRPCEvents  = cms.untracked.int32(10000)
)
