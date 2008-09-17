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

process.MuonAlignmentMonitor = cms.EDAnalyzer("MuonAlignmentAnalyzer",
    GlobalMuonTrackCollectionTag = cms.InputTag("ALCARECOMuAlZMuMu","GlobalMuon"),
    doResplots = cms.untracked.bool(True),
    # InputTags for AlCaRecoMuon format
    doSAplots = cms.untracked.bool(False),
    RecHits4DCSCCollectionTag = cms.InputTag("cscSegments"),
    resThetaRange = cms.untracked.double(0.1),
    invMassRangeMax = cms.untracked.double(200.0),
    resPhiRange = cms.untracked.double(0.1),
    min1DTrackRecHitSize = cms.untracked.uint32(1),
    doGBplots = cms.untracked.bool(False),
    invMassRangeMin = cms.untracked.double(0.0),
    #       To do resolution plots:
    #       untracked string DataType = "SimData"      # needs g4SimHits!!!
    # range of pt/mass histograms to analyze
    ptRangeMin = cms.untracked.double(0.0),
    min4DTrackSegmentSize = cms.untracked.uint32(1),
    nbins = cms.untracked.uint32(500),
    DataType = cms.untracked.string('RealData'),
    resLocalYRangeStation3 = cms.untracked.double(5.0),
    resLocalYRangeStation2 = cms.untracked.double(0.7),
    resLocalYRangeStation4 = cms.untracked.double(5.0),
    RecHits4DDTCollectionTag = cms.InputTag("dt4DSegments"),
    resLocalYRangeStation1 = cms.untracked.double(0.7),
    resLocalXRangeStation4 = cms.untracked.double(3.0),
    resLocalXRangeStation2 = cms.untracked.double(0.3),
    resLocalXRangeStation3 = cms.untracked.double(3.0),
    StandAloneTrackCollectionTag = cms.InputTag("ALCARECOMuAlZMuMu","StandAlone"),
    #residual range limits: cm and rad
    resLocalXRangeStation1 = cms.untracked.double(0.1),
    ptRangeMax = cms.untracked.double(300.0)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MuonAlignmentMonitor.root')
)

process.p = cms.Path(process.MuonAlignmentMonitor)
process.DTGeometryESModule.applyAlignment = True
process.CSCGeometryESModule.applyAlignment = True


