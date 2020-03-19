import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
#process.load("Configuration.Geometry.GeometryExtended2018_cff")
#process.load("Configuration.Geometry.GeometryExtended2026D35_cff")
#process.load("Configuration.Geometry.GeometryExtended2026D41_cff")
#process.load("Configuration.Geometry.GeometryExtended2026D46_cff")
process.load("Configuration.Geometry.GeometryExtended2026D49_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HGCalGeom')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load = cms.EDAnalyzer("OutputDDToDDL",
                              rotNumSeed = cms.int32(0),
                              fileName = cms.untracked.string("geom2026D49.xml")
)

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.load)
process.ep = cms.EndPath(process.myprint)

