# Configuration file to run CSCGeometryAsLayers
# printing table of layer information.
# Tim Cox 18.05.2009

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryAsLayers")

# Endcap Muon geometry
# ====================
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonEndcapIdealGeometryXML_cfi")

# Fake alignment is/should be ideal geometry
# ==========================================
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.fake2 = process.FakeAlignmentSource
del process.FakeAlignmentSource
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource", "fake2")


# flags for modelling of CSC layer & strip geometry
# =================================================
process.load("Geometry.CSCGeometryBuilder.cscGeometry_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# I used to append CSCGeometryESModule but this fails to activate any debug output in 310p7 on 18.05.2009.
# Must use *

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules.append('*')
process.MessageLogger.cout = cms.untracked.PSet(
   threshold = cms.untracked.string('DEBUG'),
   default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
   CSCLayerGeometry = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
   CSCGeometry = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
   CSCGeometryBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

process.producer = cms.EDAnalyzer("CSCGeometryAsLayers")

process.p1 = cms.Path(process.producer)

