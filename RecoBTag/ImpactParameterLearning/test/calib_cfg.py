# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

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
#   source = EmptySource {untracked uint32 firstRun=1 }
#untracked PSet maxEvents = {untracked int32 input = 100}
# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(  "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_1.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_10.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_11.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_12.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_14.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_15.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_16.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_17.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_18.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_19.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_2.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_20.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_21.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_22.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_24.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_25.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_26.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_27.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_28.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_29.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_3.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_30.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_4.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_5.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_6.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_7.root",
	    "rfio:/dpm/in2p3.fr/home/cms/jandrea/QCDFastSimAOD_184_80_120/QCDFastSimAOD_184_80_120_9.root"
    )
)

process.ipCalib = cms.EDFilter("ImpactParameterCalibration",
    writeToDB = cms.bool(False),
    writeToBinary = cms.bool(True),
    nBins = cms.int32(10000),
    maxSignificance = cms.double(50.0),
    writeToRootXML = cms.bool(True),
    tagInfoSrc = cms.InputTag("impactParameterTagInfos"),
    inputCategories = cms.string('HardCoded'),
    primaryVertexSrc = cms.InputTag("offlinePrimaryVertices")
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    authenticationMethod = cms.untracked.uint32(1),
    loadBlobStreamer = cms.untracked.bool(True),
    catalog = cms.untracked.string('file:mycatalog_new.xml'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:btagnew_new.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('BTagTrackProbability2DRcd'),
        tag = cms.string('probBTagPDF2D_tag')
    ), 
        cms.PSet(
            record = cms.string('BTagTrackProbability3DRcd'),
            tag = cms.string('probBTagPDF3D_tag')
        ))
)

process.p = cms.Path(process.ipCalib)

