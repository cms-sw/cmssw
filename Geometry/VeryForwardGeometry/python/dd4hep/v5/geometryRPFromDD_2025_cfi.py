import FWCore.ParameterSet.Config as cms

DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/VeryForwardGeometry/data/dd4hep/v5/geometryRPFromDD_2025.xml'),
                                            appendToDataLabel = cms.string('XMLIdealGeometryESSource_CTPPS')
)

DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                            appendToDataLabel = cms.string('XMLIdealGeometryESSource_CTPPS')
)

ctppsGeometryESModule = cms.ESProducer("CTPPSGeometryESModule",
    fromDD4hep = cms.untracked.bool(True),
    isRun2 = cms.bool(False),
    verbosity = cms.untracked.uint32(1),
    compactViewTag = cms.string('XMLIdealGeometryESSource_CTPPS')
)

