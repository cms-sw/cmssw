import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Geometry.CMSCommonData.cmsIdealGeometryDB_cff")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.source = cms.Source("EmptySource")

process.prpc = cms.EDAnalyzer("RPCGeometryAnalyzer")

process.pdt = cms.EDAnalyzer("DTGeometryAnalyzer")

process.pcsc = cms.EDAnalyzer("CSCGeometryAnalyzer")

process.ptrak = cms.EDAnalyzer("TrackerDigiGeometryAnalyzer")

process.pcalo = cms.EDAnalyzer("CaloGeometryAnalyzer")

process.myprint = cms.OutputModule("AsciiOutputModule")

process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    warning = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical', 
        'info', 
        'warning')
)

process.p1 = cms.Path(process.prpc*process.ptrak*process.pdt*process.pcsc*process.pcalo)
process.ep = cms.EndPath(process.myprint)


