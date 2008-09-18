# The following comments couldn't be translated into the new config version:

# FRONTIER

import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonAlignmentMonitor")
# Messages
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# ideal geometry
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")

# Muon alignment: from Frontier 
# it contain replaces
# Geometry to use in the extrapolator
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_DevDB_cff")

process.source = cms.Source("PoolSource",
    #AlCaReco File
    fileNames = cms.untracked.vstring('/store/mc/CSA08/MuonPT11/ALCARECO/1PB_V2_RECO_MuAlZMuMu_v1/0029/04D75FB7-E624-DD11-A822-001D09F2437B.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.muonAlignment = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTAlignmentRcd'),
        tag = cms.string('DT1InversepbScenario200v1_mc')
    ), 
        cms.PSet(
            record = cms.string('DTAlignmentErrorRcd'),
            tag = cms.string('DT1InversepbScenarioErrors200v1_mc')
        ), 
        cms.PSet(
            record = cms.string('CSCAlignmentRcd'),
            tag = cms.string('CSC1InversepbScenario200v1_mc')
        ), 
        cms.PSet(
            record = cms.string('CSCAlignmentErrorRcd'),
            tag = cms.string('CSC1InversepbScenarioErrors200v1_mc')
        )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_ALIGNMENT')
)

process.load("Alignment.OfflineValidation.MuonAlignmentAnalyzer_cfi")
# InputTags for AlCaRecoMuon format (e.g.)
process.MuonAlignmentMonitor.StandAloneTrackCollectionTag = "ALCARECOMuAlZMuMu:StandAlone"
process.MuonAlignmentMonitor.GlobalMuonTrackCollectionTag = "ALCARECOMuAlZMuMu:GlobalMuon"

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MuonAlignmentMonitor.root')
)

process.p = cms.Path(process.MuonAlignmentMonitor)
process.DTGeometryESModule.applyAlignment = True
process.CSCGeometryESModule.applyAlignment = True


