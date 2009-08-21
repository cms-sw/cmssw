import FWCore.ParameterSet.Config as cms

process = cms.Process("DBGeometryTest")
#process.load("DetectorDescription.OfflineDBLoader.test.cmsIdealGeometryForWrite_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptySource")

process.myprint = cms.OutputModule("AsciiOutputModule")

#"CMSCutsRegion" value="BeamPipeInternal"
process.prod = cms.EDAnalyzer("TestSpecParAnalyzer"
                                  , specName = cms.string("CMSCutsRegion")
                                  , specStrValue = cms.untracked.string("BeamPipeOutside")
                                  #, specDblValue = cms.untracked.double(0.0)
                              )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.prod)
process.e1 = cms.EndPath(process.myprint)
