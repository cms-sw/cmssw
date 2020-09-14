import FWCore.ParameterSet.Config as cms

process = cms.Process("DTGeometryTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load('Configuration.StandardSequences.DD4hep_GeometrySim_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")

process.MessageLogger = cms.Service(
    "MessageLogger",
    statistics = cms.untracked.vstring('cout', 'dtGeometry'),
    categories = cms.untracked.vstring('DTGeometryTest', 'Geometry'),
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
            ),
        Geometry = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
            )
        ),
    destinations = cms.untracked.vstring('cout',
                                         'dtGeometry')
    )

process.DTGeometryESProducer = cms.ESProducer("DTGeometryESProducer",
                                              DDDetector = cms.ESInputTag('',''),
                                              appendToDataLabel = cms.string(''),
                                              applyAlignment = cms.bool(False),
                                              alignmentsLabel = cms.string(''),
                                              attribute = cms.string('MuStructure'),
                                              value = cms.string('MuonBarrelDT'),
                                              fromDDD = cms.bool(True)
                                              )

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                 appendToDataLabel = cms.string('')
)

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('')
                                                     )

process.muonGeometryConstants.fromDD4Hep = True


process.test = cms.EDAnalyzer("DTGeometryTest",
                              DDDetector = cms.ESInputTag('','')
                              )

process.p = cms.Path(process.test)
