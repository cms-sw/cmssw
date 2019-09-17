import FWCore.ParameterSet.Config as cms

process = cms.Process('VALID')

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load('Configuration.StandardSequences.DD4hep_GeometrySim_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

#process.MessageLogger = cms.Service(
#    "MessageLogger",
#    statistics = cms.untracked.vstring('cout', 'muonNumbering'),
#    categories = cms.untracked.vstring('Geometry'),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('WARNING'),
#        noLineBreaks = cms.untracked.bool(True)
#        ),
#    muonNumbering = cms.untracked.PSet(
#        INFO = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#            ),
#        noLineBreaks = cms.untracked.bool(True),
#        DEBUG = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#            ),
#        WARNING = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#            ),
#        ERROR = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#            ),
#        threshold = cms.untracked.string('INFO'),
#        Geometry = cms.untracked.PSet(
#            limit = cms.untracked.int32(-1)
#            )
#        ),
#    destinations = cms.untracked.vstring('cout',
#                                         'muonNumbering')
#    )



process.RPCGeometryESProducer = cms.ESProducer("RPCGeometryESModule",
                                               DDDetector = cms.ESInputTag('',''),
                                               comp11 = cms.untracked.bool(True),
                                               attribute = cms.string('ReadOutName'),
                                               value = cms.string('MuonRPCHits'),
                                               useDDD = cms.untracked.bool(False),
                                               useDD4hep = cms.untracked.bool(True)
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
process.valid = cms.EDAnalyzer("RPCGeometryValidate",
                               infileName = cms.untracked.string('cmsRecoGeom-2021.root'),
#                              infileName = cms.untracked.string('/build/slomeo/CMSSW_11_0_X_2019-08-21-2300/src/cmsGeom10.root'),
                               outfileName = cms.untracked.string('validateRPCGeometry.root'),
                               tolerance = cms.untracked.int32(7)
                               )

process.p = cms.Path(process.valid)# process.test per 
