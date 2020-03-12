# Configuration file to run stubs/CSCGeometryOfStrips
# I hope this reads geometry from db
# Tim Cox 18.10.2012 for 61X

import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCGeometryTest")
process.load('Configuration.Geometry.GeometryExtended_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Geometry.CommonTopologies.globalTrackingGeometry_cfi')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')

process.GlobalTag.globaltag = 'MC_61_V2::All'
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.load("CondCore.DBCommon.CondDBCommon_cfi")


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

# Executable
# ==========
process.producer = cms.EDAnalyzer("CSCGeometryOfStrips")

process.p1 = cms.Path(process.producer)
process.CSCGeometryESModule.debugV = True

