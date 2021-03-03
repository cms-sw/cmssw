import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process('VALID',Run3_dd4hep)

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load('Configuration.StandardSequences.DD4hep_GeometrySim_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.CSCGeometryBuilder.cscGeometry_cfi")

process.MessageLogger = cms.Service("MessageLogger",
                                destinations = cms.untracked.vstring('myLog'),
                                myLog = cms.untracked.PSet(
                                threshold = cms.untracked.string('INFO'),
                                )
                            )

process.CSCGeometryESModule.applyAlignment = False

#
# Note: Please, download the geometry file from a location
#       specified by Fireworks/Geometry/data/download.url
#
# For example: wget http://cmsdoc.cern.ch/cms/data/CMSSW/Fireworks/Geometry/data/v4/cmsGeom10.root
#
process.valid = cms.EDAnalyzer("CSCGeometryValidate",
                               infileName = cms.untracked.string('cmsRecoGeom-2021.root'),
                               outfileName = cms.untracked.string('validateCSCGeometryDD4HEP.root'),
                               tolerance = cms.untracked.int32(7)
                               )

process.p = cms.Path(process.valid)
