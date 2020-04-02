import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
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

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet( INFO = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                               noLineBreaks = cms.untracked.bool(True),
                                                               threshold = cms.untracked.string('INFO'),
                                                               ),
                                    # For LogDebug/LogTrace output...
                                    categories = cms.untracked.vstring('DD4hep_TestMTDIdealGeometry','MTDGeom','DD4hep_TestMTDPath','DD4hep_TestMTDNumbering','DD4hep_TestMTDPosition'),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.testBTL+process.testETL)

