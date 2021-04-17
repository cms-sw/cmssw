import FWCore.ParameterSet.Config as cms

process = cms.Process("DDTIDAxialCableAlgoTest")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        enableStatistics = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    ),
    files = cms.untracked.PSet(
        testTIDAxialCableAlgo = cms.untracked.PSet(
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            ERROR = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            TIDGeom = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            WARNING = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            enableStatistics = cms.untracked.bool(True),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('INFO')
        )
    )
)

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
    confGeomXMLFiles = cms.FileInPath(
        'Geometry/TrackerCommonData/data/cms-test-ddtidaxialcable-algorithm.xml'
    ),
    appendToDataLabel = cms.string('TestTIDAxialCable')
)

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
    DDDetector = cms.ESInputTag('','TestTIDAxialCable')
)

process.p = cms.Path(process.testDump)
