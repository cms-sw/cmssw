import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMANA")

process.load('FWCore.MessageService.MessageLogger_cfi')

#process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr08v01XML_cfi')
#process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr10v01XML_cfi')
process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
process.load('Geometry.GEMGeometry.gemGeometry_cfi')
process.load('Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometryDB_cff')
process.load('Geometry.DTGeometryBuilder.idealForDigiDtGeometryDB_cff')
process.load('Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff')

# the analyzer configuration
process.load('RPCGEM.GEMValidation.GEMDigiAnalyzer_cfi')
process.GEMDigiAnalyzer.simTrackMatching.cscComparatorDigiInput = ""
process.GEMDigiAnalyzer.simTrackMatching.cscWireDigiInput = ""
process.GEMDigiAnalyzer.simTrackMatching.cscCLCTInput = ""
process.GEMDigiAnalyzer.simTrackMatching.cscALCTInput = ""
process.GEMDigiAnalyzer.simTrackMatching.cscLCTInput = ""

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'POSTLS161_V12::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("gem_digi_ana.root")
)

#dir_pt40 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT40_lpcgem/MuomGunPt40L1CSC50k_digi/82325e40d6202e6fec2dd983c477f3ca/'
#inputDir = dir_pt40

import os

#ls = os.listdir(inputDir)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #[inputDir[16:] + x for x in ls if x.endswith('root')]
    'file:out_digi.root'
    )
)

process.p    = cms.Path(process.GEMDigiAnalyzer)

