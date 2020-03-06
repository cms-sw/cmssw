import FWCore.ParameterSet.Config as cms

process = cms.Process('VALID')

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )


process.load('Configuration.StandardSequences.DD4hep_GeometrySim_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")


process.CSCGeometryESProducer = cms.ESProducer("CSCGeometryESModule",
                                               DDDetector = cms.ESInputTag('',''),
                                               applyAlignment = cms.bool(False),
                                               alignmentsLabel = cms.string(''),
                                               appendToDataLabel = cms.string(''),
                                               attribute = cms.string('MuStructure'),
                                               value = cms.string('MuonEndcapCSC'),
                                               useDDD = cms.bool(False),
                                               useDD4hep = cms.untracked.bool(True),
                                               debugV = cms.untracked.bool(False),
                                               useGangedStripsInME1a = cms.bool(False),
                                               useOnlyWiresInME1a = cms.bool(False),
                                               useRealWireGeometry = cms.bool(True),
                                               useCentreTIOffsets = cms.bool(False)
                                              )

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('') 
                                                     )

process.MuonNumberingESProducer = cms.ESProducer("MuonNumberingESProducer",
                                                 label = cms.string(''),
                                                 key = cms.string('MuonCommonNumbering')
                                                 )

process.test = cms.EDAnalyzer("DDTestMuonNumbering")

#
# Note: Please, download the geometry file from a location
#       specified by Fireworks/Geometry/data/download.url
#
# For example: wget http://cmsdoc.cern.ch/cms/data/CMSSW/Fireworks/Geometry/data/v4/cmsGeom10.root
#
process.valid = cms.EDAnalyzer("CSCGeometryValidate",
                               infileName = cms.untracked.string('cmsRecoGeom-2021.root'),
                               outfileName = cms.untracked.string('validateCSCGeometry.root'),
                               tolerance = cms.untracked.int32(7)
                               )

process.p = cms.Path(process.valid)# process.test per 
