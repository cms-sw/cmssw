import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTRPCFilter")

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.selRPC = cms.EDFilter("HLTRPCFilter", 
  rangestrips = cms.untracked.double(4.),
  rpcRecHits = cms.InputTag('rpcRecHits'),
  rpcDTPoints = cms.InputTag("rpcPointProducer","RPCDTExtrapolatedPoints"),
  rpcCSCPoints = cms.InputTag("rpcPointProducer","RPCCSCExtrapolatedPoints")
)

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring('file:/tmp/carrillo/outs/output.root')
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("selrpc")
    ),
    fileName = cms.untracked.string('/tmp/carrillo/afterfilter.root')
)

process.selrpc = cms.Path(process.selRPC)
process.outpath = cms.EndPath(process.FEVT)

