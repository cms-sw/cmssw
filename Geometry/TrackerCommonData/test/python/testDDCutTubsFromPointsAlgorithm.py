import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCutTubsFromPointsAlgoTest")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'testCutTubsFromPointsAlgo'),
    categories = cms.untracked.vstring('TrackerGeom'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
        ),
    testCutTubsFromPointsAlgo = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('DEBUG'),
        TrackerGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),

    destinations = cms.untracked.vstring('cout',
                                         'testCutTubsLog')
)



process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
    confGeomXMLFiles = cms.FileInPath(
        'Geometry/TrackerCommonData/data/cms-test-ddcuttubsfrompoints-algorithm.xml'
    ),
    appendToDataLabel = cms.string('testCutTubsFromPointsAlgo')
)

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
    DDDetector = cms.ESInputTag('','testCutTubsFromPointsAlgo')
)

process.p = cms.Path(process.testDump)
