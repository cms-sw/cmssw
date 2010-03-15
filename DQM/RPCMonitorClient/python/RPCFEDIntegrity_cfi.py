import FWCore.ParameterSet.Config as cms

rpcFEDIntegrity = cms.EDAnalyzer("RPCFEDIntegrity",
   RPCPrefixDir =  cms.untracked.string('RPC'),
   RPCRawCountsInputTag = cms.untracked.InputTag('rpcunpacker'),
   NumberOfFED = cms.untracked.int32(3),
   MinimumFEDID = cms.untracked.int32(790),
   MaximumFEDID= cms.untracked.int32(792)
 )


