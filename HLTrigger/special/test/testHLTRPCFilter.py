import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTRPCFilter")
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_3XY_V15::All"
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
#	'file:/tmp/carrillo/outs/output.root'
        '/store/data/Commissioning10/Cosmics/RECO/v3/000/127/155/005F9301-2E16-DF11-B60B-0030487CD6B4.root'
)
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring("selrpc")
    ),
    fileName = cms.untracked.string('/tmp/carrillo/afterfilter.root')
)

process.load("RecoLocalMuon.RPCRecHit.rpcPointProducer_cfi")
process.load("HLTrigger.special.hltRPCFilter_cfi")

process.selrpc = cms.Path(process.rpcPointProducer*process.hltRPCFilter)
process.outpath = cms.EndPath(process.FEVT)

