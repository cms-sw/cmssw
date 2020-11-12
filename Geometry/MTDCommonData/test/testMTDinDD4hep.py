import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('MTDUnitTest',
                                       'DD4hep_TestMTDIdealGeometry',
                                       'DD4hep_TestMTDPath',
                                       'DD4hep_TestMTDNumbering',
                                       'DD4hep_TestMTDPosition'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        FWKINFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
            ),
        DD4hep_TestMTDIdealGeometry = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        DD4hep_TestMTDPath = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        DD4hep_TestMTDNumbering = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        DD4hep_TestMTDPosition = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        noLineBreaks = cms.untracked.bool(True)
        ),
    mtdCommonDataDD4hep = cms.untracked.PSet(
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
        MTDUnitTest = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        ),
    destinations = cms.untracked.vstring('cout',
                                         'mtdCommonDataDD4hep')
)

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/MTDCommonData/data/dd4hep/cms-mtdD50-geometry.xml'),
                                            appendToDataLabel = cms.string('MTD')
)

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('MTD')
)

process.testBTL = cms.EDAnalyzer("DD4hep_TestMTDIdealGeometry",
                                 DDDetector = cms.ESInputTag('','MTD'), 
                                 ddTopNodeName = cms.untracked.string('BarrelTimingLayer'),
                                 theLayout = cms.untracked.uint32(4)
                                )

process.testETL = cms.EDAnalyzer("DD4hep_TestMTDIdealGeometry",
                                 DDDetector = cms.ESInputTag('','MTD'), 
                                 ddTopNodeName = cms.untracked.string('EndcapTimingLayer'),
                                 theLayout = cms.untracked.uint32(4)
                                )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.testBTL+process.testETL)

