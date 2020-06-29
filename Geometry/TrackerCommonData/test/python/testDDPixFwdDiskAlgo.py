import FWCore.ParameterSet.Config as cms

process = cms.Process("DDPixFwdDiskAlgoTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'testDDPixFwdDiskAlgo'),
    categories = cms.untracked.vstring('PixFwdGeom'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
        ),
    testDDPixFwdDiskAlgo = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
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
        PixFwdGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            )
        ),
    destinations = cms.untracked.vstring('cout',
                                         'testDDPixFwdDiskAlgo')
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
    confGeomXMLFiles = cms.FileInPath('Geometry/TrackerCommonData/data/cms-test-ddpixfwddisk-algo.xml'),
    appendToDataLabel = cms.string('testDDPixFwdDiskAlgo')
    )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
    DDDetector = cms.ESInputTag('','testDDPixFwdDiskAlgo')
)

process.p = cms.Path(process.testDump)
