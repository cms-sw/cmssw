import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/MTDCommonData/data/dd4hep/cms-mtd-geometry.xml'),
                                            appendToDataLabel = cms.string('MTD')
)

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('MTD')
)

#process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
#                                                 appendToDataLabel = cms.string('MTD')
#)

process.myprint = cms.OutputModule("AsciiOutputModule")

process.testBTL = cms.EDAnalyzer("DD4hep_TestMTDNumbering",
                                 DDDetector = cms.ESInputTag('','MTD'), 
                                 outFileName = cms.untracked.string('BTL'),
                                 numNodesToDump = cms.untracked.uint32(0),
                                 ddTopNodeName = cms.untracked.string('BarrelTimingLayer'),
                                 theLayout = cms.untracked.uint32(4)
                                ) 

# process.testETL = cms.EDAnalyzer("TestMTDNumbering",
#                                label = cms.untracked.string(''),
#                                isMagField = cms.untracked.bool(False),
#                                outFileName = cms.untracked.string('ETL'),
#                                numNodesToDump = cms.untracked.uint32(0),
#                                ddTopNodeName = cms.untracked.string('etl:EndcapTimingLayer')
#                                )


# process.testBTLpos = cms.EDAnalyzer("TestMTDPosition",
#                                label = cms.untracked.string(''),
#                                isMagField = cms.untracked.bool(False),
#                                outFileName = cms.untracked.string('BTLpos'),
#                                numNodesToDump = cms.untracked.uint32(0),
#                                ddTopNodeName = cms.untracked.string('btl:BarrelTimingLayer')
#                                )

# process.testETLpos = cms.EDAnalyzer("TestMTDPosition",
#                                label = cms.untracked.string(''),
#                                isMagField = cms.untracked.bool(False),
#                                outFileName = cms.untracked.string('ETLpos'),
#                                numNodesToDump = cms.untracked.uint32(0),
#                                ddTopNodeName = cms.untracked.string('etl:EndcapTimingLayer')
#                                )

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet( INFO = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
                                                               noLineBreaks = cms.untracked.bool(True),
                                                               threshold = cms.untracked.string('INFO'),
                                                               ),
                                    # For LogDebug/LogTrace output...
                                    categories = cms.untracked.vstring('DD4hep_TestMTDNumbering','MTDGeom','DD4hep_TestMTDPosition'),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.Timing = cms.Service("Timing")

#process.p1 = cms.Path(process.testBTL+process.testETL+process.testBTLpos+process.testETLpos)
process.p1 = cms.Path(process.testBTL)

process.e1 = cms.EndPath(process.myprint)
