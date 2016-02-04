# The following comments couldn't be translated into the new config version:

#include "CondCore/DBCommon/data/CondDBCommon.cfi"
#replace CondDBCommon.connect = "sqlite_file:btagnew.db"
#replace CondDBCommon.catalog = "file:mycatalog.xml"
#        es_source = PoolDBESSource {
#                                  using CondDBCommon
#                                 VPSet toGet = {
#                                   {string record = "BTagTrackProbability2DRcd"
#                                     string tag = "probBTagPDF2D_tag"    },
#                                   {string record = "BTagTrackProbability3DRcd"
#                                     string tag = "probBTagPDF3D_tag"    }
#                                    }
#                                   }

import FWCore.ParameterSet.Config as cms

process = cms.Process("analyzer")
#include "Configuration/StandardSequences/data/FakeConditions.cff"
#untracked PSet maxEvents = {untracked int32 input = 100}
# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)
process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1)
)

process.ipCalib = cms.EDAnalyzer("ImpactParameterCalibration",
    writeToDB = cms.bool(True),
    writeToBinary = cms.bool(False),
    calibFile3d = cms.FileInPath('RecoBTag/ImpactParameterLearning/test/3d.xml.new'),
    resetHistograms = cms.bool(False),
    maxSignificance = cms.double(50.0),
    writeToRootXML = cms.bool(False),
    nBins = cms.int32(10000),
    tagInfoSrc = cms.InputTag("impactParameterTagInfos"),
    calibFile2d = cms.FileInPath('RecoBTag/ImpactParameterLearning/test/2d.xml.new'),
    inputCategories = cms.string('RootXML'),
    primaryVertexSrc = cms.InputTag("offlinePrimaryVerticesFromCTFTracks")
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:btagnew_new.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('BTagTrackProbability2DRcd'),
        tag = cms.string('probBTagPDF2D_tag_mc')
    ), 
        cms.PSet(
            record = cms.string('BTagTrackProbability3DRcd'),
            tag = cms.string('probBTagPDF3D_tag_mc')
        ))
)

process.p = cms.Path(process.ipCalib)

