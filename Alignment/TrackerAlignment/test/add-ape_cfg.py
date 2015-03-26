import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# Message logger service
process.load("FWCore.MessageService.MessageLogger_cfi")

# Ideal geometry producer
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

# Database output service
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentErrorExtendedRcd'),
        tag = cms.string('TrackerNoErrors150')
    )),
    connect = cms.string('sqlite_file:TrackerIdealGeometry.db')
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.addApe = cms.EDAnalyzer("ApeAdder",
    apeVector = cms.untracked.vdouble(0.0, 0.0, 0.0)
)

process.asciiPrint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.addApe)
process.ep = cms.EndPath(process.asciiPrint)


