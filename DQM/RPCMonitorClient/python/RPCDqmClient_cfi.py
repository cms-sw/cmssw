import FWCore.ParameterSet.Config as cms

rpcdqmclient = cms.EDAnalyzer("RPCDqmClient",                               
   RPCDqmClientList = cms.untracked.vstring("RPCMultiplicityTest", "RPCDeadChannelTest", "RPCClusterSizeTest"),
   DiagnosticPrescale = cms.untracked.int32(10),
)
