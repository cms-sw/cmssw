import FWCore.ParameterSet.Config as cms

process = cms.Process('VALID')

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load('Configuration.Geometry.GeometryExtended2026D41_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.GEMGeometryBuilder.me0Geometry_cff")

#
# Note: Please, download the geometry file from a location
#       specified by Fireworks/Geometry/data/download.url
# For example: cmsRun $CMSSW_RELEASE_BASE/src/Fireworks/Geometry/python/dumpRecoGeometry_cfg.py tag=2026 version=D49
# 
#
process.valid = cms.EDAnalyzer("ME0GeometryValidate",
                               infileName = cms.untracked.string('cmsRecoGeom-2026.root'),
                               outfileName = cms.untracked.string('validateME0GeometryOldDD.root'),
                               tolerance = cms.untracked.int32(7)
                               )

process.p = cms.Path(process.valid)
