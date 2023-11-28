import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.load('FWCore.MessageLogger.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.RPCGeometry=dict()

process.test1 = cms.EDAnalyzer("RPCGEO")
process.test2 = cms.EDAnalyzer("RPCGeometryAnalyzer")

process.p = cms.Path(process.test1+process.test2)

