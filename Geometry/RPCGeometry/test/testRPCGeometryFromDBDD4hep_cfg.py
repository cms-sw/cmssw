import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("Demo", Run3_dd4hep)

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['upgrade2021']

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

