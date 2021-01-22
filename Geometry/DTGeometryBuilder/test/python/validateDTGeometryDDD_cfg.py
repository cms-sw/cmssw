import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3
 
process = cms.Process('VALID',Run3)
 
process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(1)
     )

process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.DTGeometryBuilder.dtGeometry_cfi")

process.MessageLogger = cms.Service("MessageLogger",
                                destinations = cms.untracked.vstring('myLog'),
                                myLog = cms.untracked.PSet(
                                threshold = cms.untracked.string('INFO'),
                                )
                            )

process.valid = cms.EDAnalyzer("DTGeometryValidate",
                                infileName = cms.untracked.string('cmsRecoGeom-2021.root'),
                                outfileName = cms.untracked.string('validateDTGeometryDDD.root'),
                                )
 
process.p = cms.Path(process.valid)
