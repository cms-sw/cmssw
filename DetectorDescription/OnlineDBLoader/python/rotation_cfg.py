import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.ppt = cms.EDAnalyzer("RotationForOnline")

process.printM = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.ppt)
process.ep = cms.EndPath(process.printM)

