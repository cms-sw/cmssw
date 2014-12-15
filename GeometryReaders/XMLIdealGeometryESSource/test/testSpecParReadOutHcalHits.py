import FWCore.ParameterSet.Config as cms

process = cms.Process("DBGeometryTest")
process.load('Geometry.CMSCommonData.cmsIdealGeometry2015XML_cfi')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptySource")

process.myprint = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("TestSpecParAnalyzer",
                              specName = cms.string('ReadOutName'),
                              specStrValue = cms.untracked.string('HcalHits')
                              )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.prod)
process.e1 = cms.EndPath(process.myprint)
