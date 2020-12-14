import FWCore.ParameterSet.Config as cms

process = cms.Process('VALID')

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load('Configuration.StandardSequences.DD4hep_GeometrySim_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")

process.MessageLogger = cms.Service("MessageLogger",
                                destinations = cms.untracked.vstring('myLog'),
                                myLog = cms.untracked.PSet(
                                threshold = cms.untracked.string('INFO'),
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


process.valid = cms.EDAnalyzer("DTGeometryValidate",

                               infileName = cms.untracked.string('Geometry/DTGeometryBuilder/data/cmsRecoGeom-2021.root'),
                               outfileName = cms.untracked.string('validateDTGeometry.root'),
                               tolerance = cms.untracked.int32(7)
                               )

process.p = cms.Path(process.valid)
