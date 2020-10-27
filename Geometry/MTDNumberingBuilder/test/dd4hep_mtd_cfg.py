import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("GeometryTest",dd4hep)

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('MTDUnitTest',
                                       'GeometricTimingDetAnalyzer',
                                       'MTDTopologyAnalyzer'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        GeometricTimingDetAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        MTDTopologyAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            ),
        noLineBreaks = cms.untracked.bool(True)
        ),
    mtdNumberingDD4hep = cms.untracked.PSet(
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
                                         'mtdNumberingDD4hep')
)

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/MTDCommonData/data/dd4hep/cms-mtdD50-geometry.xml'),
                                            appendToDataLabel = cms.string('')
)

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                 appendToDataLabel = cms.string('')
)

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('')
)

process.load("Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff")

process.load("Geometry.MTDNumberingBuilder.mtdTopology_cfi")
process.load("Geometry.MTDGeometryBuilder.mtdParameters_cff")

process.prod = cms.EDAnalyzer("GeometricTimingDetAnalyzer")

process.prod1 = cms.EDAnalyzer("DD4hep_MTDTopologyAnalyzer",
                               DDDetector = cms.ESInputTag('',''),
)

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.prod+process.prod1)
