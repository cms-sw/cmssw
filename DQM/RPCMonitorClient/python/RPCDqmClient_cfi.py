import FWCore.ParameterSet.Config as cms

rpcdqmclient = cms.EDAnalyzer("RPCDqmClient",                               
                              RPCDqmClientList = cms.untracked.vstring("RPCMultiplicityTest", "RPCDeadChannelTest", "RPCClusterSizeTest"),
                              DiagnosticGlobalPrescale = cms.untracked.int32(1),
                              MinimumRPCEvents =  cms.untracked.int32(1)
                              )
