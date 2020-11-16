import FWCore.ParameterSet.Config as cms

process = cms.Process("MTDGeometryTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'mtdGeometry'),
    categories = cms.untracked.vstring('Geometry'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
        ),
    mtdGeometry = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        FWKINFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        threshold = cms.untracked.string('INFO'),
        Geometry = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            )
        ),
    destinations = cms.untracked.vstring('cout',
                                         'mtdGeometry')
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/MTDCommonData/data/dd4hep/cms-mtdD50-geometry.xml'),
                                            appendToDataLabel = cms.string('MTD')
                                            )

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('MTD')
                                                     )

process.DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                                    appendToDataLabel = cms.string('MTD')
                                                    )

process.test = cms.EDAnalyzer("DDCMSDetector",
                              DDDetector = cms.ESInputTag('','MTD')
                              )

process.dump = cms.EDAnalyzer("DDTestDumpFile",
                              DDDetector = cms.ESInputTag('','MTD')
                              )

process.p = cms.Path(process.test+process.dump)
