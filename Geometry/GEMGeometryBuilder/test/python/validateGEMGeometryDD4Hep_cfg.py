import FWCore.ParameterSet.Config as cms

process = cms.Process('VALID')

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load('Configuration.StandardSequences.DD4hep_GeometrySim_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.GEMGeometryBuilder.gemGeometry_cff")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.Geometry=dict()
    process.MessageLogger.GEMNumberingScheme=dict()


process.test = cms.EDAnalyzer("DDTestMuonNumbering")

#
# Note: Please, download the geometry file from a location
#       specified by Fireworks/Geometry/data/download.url
#
# For example: cmsRun $CMSSW_RELEASE_BASE/src/Fireworks/Geometry/python/dumpRecoGeometry_cfg.py tag=2021
#
process.valid = cms.EDAnalyzer("GEMGeometryValidate",
                               infileName = cms.untracked.string('cmsRecoGeom-2021.root'),
                               outfileName = cms.untracked.string('validateGEMGeometry.root'),
                               tolerance = cms.untracked.int32(7)
                               )

process.muonGeometryConstants.fromDD4Hep = True
process.gemGeometry.fromDDD = False
process.gemGeometry.fromDD4Hep = True

process.p = cms.Path(process.valid)
