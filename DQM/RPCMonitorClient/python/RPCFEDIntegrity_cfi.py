import FWCore.ParameterSet.Config as cms

rpcFEDIntegrity = cms.EDAnalyzer("RPCFEDIntegrity",
   RPCPrefixDir =  cms.untracked.string('RPC'),
   NumberOfFED = cms.untracked.int32(3),
   MinimumFEDID = cms.untracked.int32(790),
   MaximumFEDID= cms.untracked.int32(792)
 )


rpcRawDataCount = cms.EDAnalyzer("RPCMonitorRaw",
  writeHistograms = cms.untracked.bool(False),
  histoFileName = cms.untracked.string('rpcMonitorRaw.root')
)

rpcDATAIntegrity= cms.Sequence(rpcFEDIntegrity * rpcRawDataCount)
