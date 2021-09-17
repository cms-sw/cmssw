import FWCore.ParameterSet.Config as cms

process = cms.Process("DTGeometryTest")
process.add_(cms.Service("InitRootHandlers", ResetRootErrHandler = cms.untracked.bool(False)))
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

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
        dumpGeometry = cms.untracked.PSet(
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            ERROR = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            Geometry = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
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
                                            confGeomXMLFiles = cms.FileInPath('Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2021.xml'),
                                            appendToDataLabel = cms.string('MUON')
                                            )

process.DTGeometryESProducer = cms.ESProducer("DTGeometryESProducer",
                                              DDDetector = cms.ESInputTag('','MUON'),
                                              appendToDataLabel = cms.string(''),
                                              applyAlignment = cms.bool(False),
                                              alignmentsLabel = cms.string(''),
                                              attribute = cms.string('MuStructure'),
                                              value = cms.string('MuonBarrelDT'),
                                              fromDDD = cms.bool(True)
                                              )

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('MUON')
                                                     )

process.MuonNumberingESProducer = cms.ESProducer("MuonNumberingESProducer",
                                                 label = cms.string('MUON'),
                                                 key = cms.string('MuonCommonNumbering')
                                                 )

process.test = cms.EDAnalyzer("DTGeometryTest",
                              DDDetector = cms.ESInputTag('','MUON')
                              )

process.dump = cms.EDAnalyzer("DDTestDumpGeometry",
                              DDDetector = cms.ESInputTag('','MUON')
                              )

process.p = cms.Path(process.test+process.dump)
