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

process.testBTL = cms.EDAnalyzer("DD4hep_TestMTDNumbering",
                                 DDDetector = cms.ESInputTag('','MTD'), 
                                 outFileName = cms.untracked.string('BTL'),
                                 ddTopNodeName = cms.untracked.string('BarrelTimingLayer'),
                                 theLayout = cms.untracked.uint32(4)
                                ) 

process.testETL = cms.EDAnalyzer("DD4hep_TestMTDNumbering",
                                 DDDetector = cms.ESInputTag('','MTD'), 
                                 outFileName = cms.untracked.string('ETL'),
                                 ddTopNodeName = cms.untracked.string('EndcapTimingLayer'),
                               )


process.testBTLpos = cms.EDAnalyzer("DD4hep_TestMTDPosition",
                                    DDDetector = cms.ESInputTag('','MTD'), 
                                    outFileName = cms.untracked.string('BTLpos'),
                                    ddTopNodeName = cms.untracked.string('BarrelTimingLayer')
                                )

process.testETLpos = cms.EDAnalyzer("DD4hep_TestMTDPosition",
                                    DDDetector = cms.ESInputTag('','MTD'), 
                                    outFileName = cms.untracked.string('ETLpos'),
                                    ddTopNodeName = cms.untracked.string('EndcapTimingLayer')
                               )

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

process.p1 = cms.Path(process.testBTL+process.testETL+process.testBTLpos+process.testETLpos)

