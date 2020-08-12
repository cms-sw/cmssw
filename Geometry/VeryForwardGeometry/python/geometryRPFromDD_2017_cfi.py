import FWCore.ParameterSet.Config as cms

DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/VeryForwardGeometry/data/dd4hep/geometryRPFromDD_2017.xml'),
                                            appendToDataLabel = cms.string('CMS')
)

ctppsGeometryESModule = cms.ESProducer("PPSGeometryESProducer",
    verbosity = cms.untracked.uint32(1),
    detectorTag = cms.string("CMS")
)
