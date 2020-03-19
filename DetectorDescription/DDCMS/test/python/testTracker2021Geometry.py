import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'dd4hepTrackerTest'),
    categories = cms.untracked.vstring('TECGeom'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        noLineBreaks = cms.untracked.bool(True)
    ),

    dd4hepTrackerTest = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('DEBUG'),
        TECGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),

    destinations = cms.untracked.vstring('cout',
                                         'dd4hepTrackerTest')
)




process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-tracker-2021.xml'),
                                            appendToDataLabel = cms.string('CMS')
                                            )

process.DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                                    appendToDataLabel = cms.string('CMS')
                                                    )

process.test = cms.EDAnalyzer("DDCMSDetector",
                              DDDetector = cms.ESInputTag('','CMS')
                              )

process.testVectors = cms.EDAnalyzer("DDTestVectors",
                                     DDDetector = cms.ESInputTag('','CMS')
                                     )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  DDDetector = cms.ESInputTag('','CMS')
                              )

process.testGeoIter = cms.EDAnalyzer("DDTestDumpGeometry",
                                     DDDetector = cms.ESInputTag('','CMS')
                                     )

process.p = cms.Path(
    process.test
    +process.testVectors
    +process.testDump
    +process.testGeoIter)
