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
    fileNames = cms.untracked.vstring('file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_49.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_4.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_50.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_51.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_52.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_53.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_54.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_55.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_56.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_57.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_58.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_59.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_5.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_60.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_61.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_65.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_66.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_68.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_6.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_71.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_72.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_73.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_74.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_75.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_77.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_78.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_7.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_80.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_83.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_86.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_88.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_89.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_8.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_90.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_91.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_92.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_93.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_94.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_95.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_96.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_97.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_98.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_99.root', 
        'file:/export/data1/rizzi/qcd1.6.0/qcd5080pf/btagCalibBTAGCALB_9.root')
)

process.ipCalib = cms.EDFilter("ImpactParameterCalibration",
    writeToDB = cms.bool(False),
    writeToBinary = cms.bool(True),
    nBins = cms.int32(10000),
    maxSignificance = cms.double(50.0),
    writeToRootXML = cms.bool(True),
    tagInfoSrc = cms.InputTag("impactParameterTagInfos"),
    inputCategories = cms.string('HardCoded'),
    primaryVertexSrc = cms.InputTag("offlinePrimaryVerticesFromCTFTracks")
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    authenticationMethod = cms.untracked.uint32(1),
    loadBlobStreamer = cms.untracked.bool(True),
    catalog = cms.untracked.string('file:mycatalog_new.xml'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    timetype = cms.string('runnumber'),
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

