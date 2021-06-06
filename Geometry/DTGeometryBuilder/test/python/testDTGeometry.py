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
        dtGeometry = cms.untracked.PSet(
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            DTGeometryTest = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
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
