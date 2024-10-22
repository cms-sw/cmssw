# Configuration file to run CSCLayerGeometryInside
# 
# Tim Cox 06.05.2009

import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCGeometryTest")

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
process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# root output file
# ================
process.TFileService = cms.Service("TFileService", 
      fileName = cms.string("clgi.root"),
      closeFileFast = cms.untracked.bool(True)
)

# Configure the actual EDAnalyzer
# ===============================
process.analyze = cms.EDAnalyzer(
    "CSCLayerGeometryInside",
    ntries = cms.untracked.int32(200000),
    nbx = cms.untracked.int32(400),
    nby = cms.untracked.int32(800),
    xlo = cms.untracked.double(-50.),
    xhi = cms.untracked.double(50.),
    ylo = cms.untracked.double(-100.),
    yhi = cms.untracked.double(100.)
)

# Message Logger - default is all my debug messages off
# ==============
# Note that this has no effect on errors from DDD - they come even if this is removed!
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules.append('CSCGeometryESModule')
process.MessageLogger.cout = cms.untracked.PSet(
   threshold = cms.untracked.string('DEBUG'),
   default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
   CSCGeometry = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
   CSCGeometryBuilder = cms.untracked.PSet( limit = cms.untracked.int32(0) )
)

process.p1 = cms.Path(process.analyze)

