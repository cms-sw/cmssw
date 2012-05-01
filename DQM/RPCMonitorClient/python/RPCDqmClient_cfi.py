import FWCore.ParameterSet.Config as cms

rpcdqmclient = cms.EDAnalyzer("RPCDqmClient",                               
                              RPCDqmClientList = cms.untracked.vstring("RPCMultiplicityTest", "RPCDeadChannelTest", "RPCClusterSizeTest", "RPCOccupancyTest","RPCNoisyStripTest"),
                              DiagnosticPrescale = cms.untracked.int32(5),
                              MinimumRPCEvents  = cms.untracked.int32(10000),
                              RecHitTypeFolder = cms.untracked.string("Noise"),
                              OfflineDQM = cms.untracked.bool(True),
                              UseRollInfo = cms.untracked.bool(False),
                              EnableRPCDqmClient  = cms.untracked.bool(True)
                              )


rpcdqmMuonclient = cms.EDAnalyzer("RPCDqmClient",                               
                                  RPCDqmClientList = cms.untracked.vstring("RPCMultiplicityTest", "RPCDeadChannelTest", "RPCClusterSizeTest", "RPCOccupancyTest","RPCNoisyStripTest"),
                                  DiagnosticPrescale = cms.untracked.int32(5),
                                  MinimumRPCEvents  = cms.untracked.int32(10000),
                                  RecHitTypeFolder = cms.untracked.string("Muon"),
                                  OfflineDQM = cms.untracked.bool(True),
                                  UseRollInfo = cms.untracked.bool(False),
                                  EnableRPCDqmClient  = cms.untracked.bool(True)
                              )
