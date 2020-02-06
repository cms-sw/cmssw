import FWCore.ParameterSet.Config as cms

process = cms.Process('VALID')

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load('Configuration.StandardSequences.DD4hep_GeometrySim_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")


process.GEMGeometryESProducer = cms.ESProducer("GEMGeometryESModule",
                                               DDDetector = cms.ESInputTag('',''),
                                               applyAlignment = cms.bool(False),
                                               alignmentsLabel = cms.string(''),
                                               attribute = cms.string('MuStructure'),
                                               value = cms.string('MuonEndCapGEM'),
                                               useDDD = cms.bool(False),
                                               useDD4hep = cms.untracked.bool(True)
                                              )

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('') #it was ''
                                                     )

process.MuonNumberingESProducer = cms.ESProducer("MuonNumberingESProducer",
                                                 label = cms.string(''),# it was ''
                                                 key = cms.string('MuonCommonNumbering')
                                                 )

process.test = cms.EDAnalyzer("DDTestMuonNumbering")

#
# Note: Please, download the geometry file from a location
#       specified by Fireworks/Geometry/data/download.url
#
# For example: cmsRun $CMSSW_RELEASE_BASE/src/Fireworks/Geometry/python/dumpRecoGeometry_cfg.py tag=2021
#
process.valid = cms.EDAnalyzer("GEMGeometryValidate",
                               infileName = cms.untracked.string('cmsRecoGeom-2021.root'),
                               outfileName = cms.untracked.string('validateGEMGeometry.root'),
                               tolerance = cms.untracked.int32(7)
                               )

process.p = cms.Path(process.valid)
