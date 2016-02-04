import FWCore.ParameterSet.Config as cms

process = cms.Process("DISPLAY")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.source = cms.Source("EmptySource")

process.add_( cms.ESProducer("TGeoMgrFromDdd") )

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )
process.dump = cms.EDAnalyzer("DisplayGeom",
                                verbose = cms.untracked.bool(False),
                                level = cms.untracked.int32(11)
                            )

process.p = cms.Path(process.dump)
