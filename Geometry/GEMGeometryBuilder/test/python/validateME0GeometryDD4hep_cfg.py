import FWCore.ParameterSet.Config as cms

process = cms.Process('VALID')

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load('Configuration.StandardSequences.DD4hep_GeometrySimPhase2_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.GEMGeometryBuilder.me0Geometry_cff")

process.test = cms.EDAnalyzer("DDTestMuonNumbering")

#
# Note: Please, download the geometry file from a location
#       specified by Fireworks/Geometry/data/download.url
#
# For example: cmsRun $CMSSW_RELEASE_BASE/src/Fireworks/Geometry/python/dumpRecoGeometry_cfg.py tag=2026 version=D41
#
process.valid = cms.EDAnalyzer("ME0GeometryValidate",
                               infileName = cms.untracked.string('cmsRecoGeom-2026.root'),
                               outfileName = cms.untracked.string('validateME0GeometryDD4hep.root'),
                               tolerance = cms.untracked.int32(7)
                               )

process.muonGeometryConstants.fromDD4Hep = True
process.me0Geometry.fromDDD = False
process.me0Geometry.fromDD4Hep = True

process.p = cms.Path(process.valid)
