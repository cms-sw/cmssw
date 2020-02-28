import FWCore.ParameterSet.Config as cms

process = cms.Process("DTGeometryTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'dtGeometry'),
    categories = cms.untracked.vstring('DTGeometryTest'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noLineBreaks = cms.untracked.bool(True)
        ),
    dtGeometry = cms.untracked.PSet(
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
        DTGeometryTest = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            )
        ),
    destinations = cms.untracked.vstring('cout',
                                         'dtGeometry')
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml'),
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

process.p = cms.Path(process.test)
