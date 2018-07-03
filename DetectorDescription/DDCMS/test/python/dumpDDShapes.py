import FWCore.ParameterSet.Config as cms

process = cms.Process("SHAPESDUMP")
process.source = cms.Source("EmptySource")
process.load("DetectorDescription.DDCMS.cmsShapesXML_cfi")
##process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.add_(cms.ESProducer("TGeoMgrFromDdd",
                            verbose = cms.untracked.bool(False),
                            level = cms.untracked.int32(14)
                            ))

process.dump = cms.EDAnalyzer("DumpSimGeometry", 
                              tag = cms.untracked.string("DDShapes"),
                              outputFileName = cms.untracked.string("cmsDDShapesTest.root"))

process.p = cms.Path(process.dump)
