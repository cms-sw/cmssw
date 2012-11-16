import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load('Configuration.Geometry.GeometryExtendedPostLS2_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
process.load('Geometry.GEMGeometry.gemGeometry_cfi')

process.load('FWCore.MessageLogger.MessageLogger_cfi')
#process.MessageLogger = cms.Service("MessageLogger",
#                                    destinations = cms.untracked.vstring("detailedInfo","critical"),
#                                    detailedInfo = cms.untracked.PSet(
#                                                   threshold = cms.untracked.string("DEBUG")),
#                                    debugModules = cms.untracked.vstring("GEMGeometryAnalyzer","GEMGeometryBuilderFromDDD")
#)

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger = cms.Service("MessageLogger")

process.test2 = cms.EDAnalyzer("GEMGeometryAnalyzer")

#process.p = cms.Path(process.test1+process.test2)
process.p = cms.Path(process.test2)

