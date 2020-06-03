import FWCore.ParameterSet.Config as cms

process = cms.Process("DDPixBarLayerUpgradeAlgoTest")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'testDDPixBarLayerUpgradeAlgo'),
    categories = cms.untracked.vstring('PixelGeom'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
    ),

    testDDPixBarLayerUpgradeAlgo = cms.untracked.PSet(
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
        PixelGeom = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),

    destinations = cms.untracked.vstring('cout',
                                         'testDDPixBarLayerUpgradeAlgo')
)


process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
    confGeomXMLFiles = cms.FileInPath(
        'Geometry/TrackerCommonData/data/cms-test-ddpixbarlayerupgrade-algorithm.xml'
    ),
    appendToDataLabel = cms.string('testPixBarLayerUpgradeAlgo')
)

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
    DDDetector = cms.ESInputTag('','testPixBarLayerUpgradeAlgo')
)

process.p = cms.Path(process.testDump)
