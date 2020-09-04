import FWCore.ParameterSet.Config as cms

DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/VeryForwardGeometry/data/dd4hep/empty.xml'),
                                            label = cms.string('CTPPS'),
                                            fromDB = cms.bool(True),
                                            appendToDataLabel = cms.string('XMLIdealGeometryESSource_CTPPS')
)


DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                            appendToDataLabel = cms.string('XMLIdealGeometryESSource_CTPPS')
)                                


ctppsGeometryESModule = cms.ESProducer("PPSGeometryESProducer",
    verbosity = cms.untracked.uint32(1),
    detectorTag = cms.string("XMLIdealGeometryESSource_CTPPS")
)

