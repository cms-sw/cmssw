import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
#process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr08v01XML_cfi')
#process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr10v01XML_cfi')
process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
process.load('Geometry.GEMGeometry.gemGeometry_cfi')

process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')

# the analyzer configuration
process.load('RPCGEM.GEMValidation.GEMSimHitAnalyzer_cfi')
process.GEMSimHitAnalyzer.simTrackMatching.gemDigiInput = ""
process.GEMSimHitAnalyzer.simTrackMatching.gemPadDigiInput = ""
process.GEMSimHitAnalyzer.simTrackMatching.gemCoPadDigiInput = ""
process.GEMSimHitAnalyzer.simTrackMatching.cscComparatorDigiInput = ""
process.GEMSimHitAnalyzer.simTrackMatching.cscWireDigiInput = ""
process.GEMSimHitAnalyzer.simTrackMatching.cscCLCTInput = ""
process.GEMSimHitAnalyzer.simTrackMatching.cscALCTInput = ""
process.GEMSimHitAnalyzer.simTrackMatching.cscLCTInput = ""

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'POSTLS161_V12::All'
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

## ff = open('filelist_minbias_61M.txt', "r")
## pu_files = ff.read().split('\n')
## ff.close()
## pu_files = filter(lambda x: x.endswith('.root'),  pu_files)

dir_pt40 = '/pnfs/cms/WAX/11/store/user/lpcgem/willhf/willhf/muonGun_50k_pT40_lpcgem/muonGun_50k_pT40_lpcgem/c25a99a3a5d3061319c1beac698b55b1/'

inputDir = dir_pt40

import os

ls = os.listdir(inputDir)

process.source = cms.Source("PoolSource",
##    fileNames = cms.untracked.vstring(*pu_files)
  fileNames = cms.untracked.vstring(
    [inputDir[16:] + x for x in ls if x.endswith('root')]
  )
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string("gem_sh_ana.test.root")
)

process.p = cms.Path(process.GEMSimHitAnalyzer)

