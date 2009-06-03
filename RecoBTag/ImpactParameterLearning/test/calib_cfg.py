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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( 
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0003/78F7179F-FDC4-DD11-AC0A-000423D986A8.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0003/26AB4F4E-F6C4-DD11-A590-000423D94E1C.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/F2A14CDA-D3C3-DD11-98D8-001617C3B654.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/EE8A920A-D3C3-DD11-A27C-0019DB29C614.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/E2A3FA7A-D4C3-DD11-8EBD-000423D6CA72.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/E0494347-DAC3-DD11-B0D0-000423D986A8.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/CE846D00-D3C3-DD11-86D9-001617C3B66C.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/CC6CBB0B-D5C3-DD11-994D-001617DBD556.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/CA94737C-DBC3-DD11-8333-000423D99660.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/CA1DE0E4-F5C3-DD11-BCFA-001617E30D06.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/BEA0916D-F4C3-DD11-B972-000423D6006E.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/8E0C0DDB-DEC3-DD11-AC37-000423D6CA6E.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/8614520B-ECC3-DD11-8C12-000423D9870C.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/78BF379C-E7C3-DD11-A131-000423D6B48C.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/68493222-D9C3-DD11-83F0-000423D95220.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/560EED0E-E3C3-DD11-ACE8-001617E30CD4.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/4CBB799F-D7C3-DD11-8D56-000423D6CAF2.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/2C362288-D2C3-DD11-AB5E-000423D98EC4.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/2A015C80-F0C3-DD11-AF8C-000423D986C4.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/1E542E1B-DDC3-DD11-8224-001617C3B6C6.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/18C00976-E1C3-DD11-9C7B-001617DBCF6A.root',
        '/store/relval/CMSSW_2_2_1/RelValQCD_Pt_50_80/GEN-SIM-RECO/IDEAL_V9_v1/0001/0A9B4208-D1C3-DD11-BB64-001617C3B77C.root' 
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

