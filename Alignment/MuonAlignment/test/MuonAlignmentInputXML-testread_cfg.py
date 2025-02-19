import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.MuonGeometryDBConverter = cms.EDAnalyzer("MuonGeometryDBConverter",
    outputXML = cms.PSet(
        relativeto = cms.string('ideal'),
        eulerAngles = cms.bool(False),
        survey = cms.bool(False),
        rawIds = cms.bool(False),
        fileName = cms.string('tmp2.xml'),
        precision = cms.int32(6)
    ),
    shiftErr = cms.double(5.0),
    fileName = cms.string('tmp.xml'),
    input = cms.string('xml'),
    output = cms.string('xml'),
    angleErr = cms.double(1.0)
)

process.p = cms.Path(process.MuonGeometryDBConverter)


