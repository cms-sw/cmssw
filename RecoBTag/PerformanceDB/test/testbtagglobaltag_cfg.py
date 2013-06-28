import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load ("RecoBTag.PerformanceDB.BTagPerformanceDBWinter13")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")
process.demo = cms.EDAnalyzer('TestBtagPayloads')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "START53_V24::All"
process.GlobalTag.connect = 'sqlite_file:/afs/cern.ch/user/a/alcaprod/public/Alca/GlobalTag/START53_V24.db'

process.p = cms.Path(process.demo)
