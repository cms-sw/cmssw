# Configuration file to run stubs/CSCGeometryAnalyser

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules.append('CSCGeometryESModule')
process.MessageLogger.cout = cms.untracked.PSet(
       threshold = cms.untracked.string('DEBUG'),
       default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
       CSCGeometry = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
       CSCGeometryBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

process.producer = cms.EDAnalyzer("CSCGeometryAnalyzer")

process.p1 = cms.Path(process.producer)
process.CSCGeometryESModule.debugV = True

