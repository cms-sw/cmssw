# Configuration file to run CSCGeometryAnalyzer

import FWCore.ParameterSet.Config as cms

process = cms.Process('CSCGeometryAnalyzer')

# Endcap Muon geometry
# ====================
process.load('Configuration.Geometry.GeometryExtended_cff')

# Fake alignment is/should be ideal geometry
# ==========================================
process.load('Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi')
process.preferFakeAlign = cms.ESPrefer('FakeAlignmentSource')

# flags for modelling of CSC layer & strip geometry
# =================================================
process.load('Geometry.CSCGeometry.cscGeometry_cfi')

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.debugModules.append('CSCGeometryESModule')
process.MessageLogger.cout = cms.untracked.PSet(
   threshold = cms.untracked.string('DEBUG'),
   default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
   CSCGeometry = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
   CSCGeometryBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

process.producer = cms.EDAnalyzer('CSCGeometryAnalyzer')

process.p1 = cms.Path(process.producer)
