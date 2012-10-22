import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load ("RecoBTag.PerformanceDB.PoolBTagPerformanceDB062012")
process.load ("RecoBTag.PerformanceDB.BTagPerformanceDB062012")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")
process.demo = cms.EDAnalyzer('TestBtagPayloads')

process.p = cms.Path(process.demo)
