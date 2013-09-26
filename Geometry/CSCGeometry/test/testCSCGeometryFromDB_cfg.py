# Configuration file to run stubs/CSCGeometryAnalyser

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.XMLFromDBSource.label = cms.string('Extended')
process.GlobalTag.globaltag = 'PRE_MC62_V8::All'

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules.append('CSCGeometryESModule')
process.MessageLogger.categories.append('CSCGeometry')
process.MessageLogger.categories.append('CSCGeometryBuilder')
process.MessageLogger.cout = cms.untracked.PSet(
       threshold = cms.untracked.string('DEBUG'),
       default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
       CSCGeometry = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
       CSCGeometryBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

process.producer = cms.EDAnalyzer("CSCGeometryAnalyzer")

process.p1 = cms.Path(process.producer)
process.CSCGeometryESModule.debugV = True

