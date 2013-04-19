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


# GEM-CSC trigger pad digi producer
process.load('SimMuon.GEMDigitizer.muonGEMCSCPadDigis_cfi')

# the analyzer configuration
process.load('RPCGEM.GEMValidation.GEMCSCAnalyzer_cfi')
#process.GEMCSCAnalyzer.verbose = 2
#process.GEMCSCAnalyzer.ntupleTrackChamberDelta = False
#process.GEMCSCAnalyzer.ntupleTrackEff = True
process.GEMCSCAnalyzer.ntupleTrackChamberDelta = True
process.GEMCSCAnalyzer.ntupleTrackEff = True
process.GEMCSCAnalyzer.minPt = 1.5
#process.GEMCSCAnalyzer.simTrackMatching.verboseSimHit = 1
#process.GEMCSCAnalyzer.simTrackMatching.verboseGEMDigi = 1
#process.GEMCSCAnalyzer.simTrackMatching.verboseCSCDigi = 1
#process.GEMCSCAnalyzer.simTrackMatching.verboseCSCStub = 1
#process.GEMCSCAnalyzer.simTrackMatching.simMuOnlyGEM = False
#process.GEMCSCAnalyzer.simTrackMatching.simMuOnlyCSC = False
#process.GEMCSCAnalyzer.simTrackMatching.discardEleHitsCSC = False
#process.GEMCSCAnalyzer.simTrackMatching.discardEleHitsGEM = False


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'POSTLS161_V12::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100000) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )


dirPt5 = '/afs/cern.ch/cms/MUON/gem/muonGun_50k_pT20_digi_v3/'
dirPt20 = '/afs/cern.ch/cms/MUON/gem/muonGun_50k_pT5_digi_v2/'

dirPt5 = '/pnfs/cms/WAX/11/store/user/lpcgem/dildick/dildick/pT5_1M_v1/Digi+L1CSC-MuonGunPt5_1M/82325e40d6202e6fec2dd983c477f3ca/'
dirPt20 = '/pnfs/cms/WAX/11/store/user/lpcgem/dildick/dildick/pT20_1M_v1/Digi+L1CSC-MuonGunPt20_1M/82325e40d6202e6fec2dd983c477f3ca/'

dirPt5Pt40 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/MuomGUN_SIM_Pt5-40_50k/MuomGun_digi_Pt5-40_L1CSC_50k/82325e40d6202e6fec2dd983c477f3ca/'
dirPt2Pt50 = '/pnfs/cms/WAX/11/store/user/lpcgem/khotilov/khotilov/MuomGUN_SIM_Pt2-50_100k/MuonGun_DIGI_L1_Pt2-50_100k/29891ddb18281fff4c42a6e5f5d4bc55/'

dir_pt5 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT5_lpcgem/MuomGunPtL1CSC50k5_digi/82325e40d6202e6fec2dd983c477f3ca/'
dir_pt10 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT10_lpcgem/MuomGunPt10L1CSC50k_digi/82325e40d6202e6fec2dd983c477f3ca/'
dir_pt15 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT15_lpcgem/MuomGun_Pt15_L1CSC_50k_digi/82325e40d6202e6fec2dd983c477f3ca/'
dir_pt20 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT20_lpcgem/MuomGunPt20L1CSC50k_digi/82325e40d6202e6fec2dd983c477f3ca/'
dir_pt30 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/MuonGun_Sim_50k_pT30_v2/MuomGun_Pt30_L1CSC_50k_digi/82325e40d6202e6fec2dd983c477f3ca/'
dir_pt40 = '/pnfs/cms/WAX/11/store/user/lpcgem/yasser1/yasser/muonGun_50k_pT40_lpcgem/MuomGunPt40L1CSC50k_digi/82325e40d6202e6fec2dd983c477f3ca/'



import os

inputDir = dirPt20 ; ntupleFile = 'gem_csc_delta_pt20_or16.root'
inputDir = dirPt5  ; ntupleFile = 'gem_csc_delta_pt5_or16.root'

#inputDir = dirPt20 ; ntupleFile = 'gem_csc_delta_pt20_or8.root'
#inputDir = dirPt5  ; ntupleFile = 'gem_csc_delta_pt5_or8.root'

#inputDir = dirPt20 ; ntupleFile = 'gem_csc_delta_pt20_or4_lct.root'
#inputDir = dirPt5  ; ntupleFile = 'gem_csc_delta_pt5_or4_lct.root'

#inputDir = dirPt5Pt40  ; ntupleFile = 'gem_csc_delta_pt5pt40_or4.root'
inputDir = dirPt5Pt40  ; ntupleFile = 'gem_csc_eff_pt5pt40_or4.root'


inputDir = dir_pt5  ; ntupleFile = 'gem_csc_delta_pt5_pad4.root'
inputDir = dir_pt10  ; ntupleFile = 'gem_csc_delta_pt10_pad4.root'
inputDir = dir_pt15  ; ntupleFile = 'gem_csc_delta_pt15_pad4.root'
inputDir = dir_pt20  ; ntupleFile = 'gem_csc_delta_pt20_pad4.root'
inputDir = dir_pt30  ; ntupleFile = 'gem_csc_delta_pt30_pad4.root'
inputDir = dir_pt40  ; ntupleFile = 'gem_csc_delta_pt40_pad4.root'

#inputDir = dirPt2Pt50  ; ntupleFile = 'gem_csc_eff_pt2pt50_pad4.root'


ls = os.listdir(inputDir)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'file:out_sim.root'
#        'file:output_SingleMuPt40.root'
#    'file:/afs/cern.ch/cms/MUON/gem/SingleMuPt40Fwd/SingleMuPt40Fwd_20121205_FixedGeometry_DIGI.root'
     #['file:'+inputDir+x for x in ls if x.endswith('root')]
     [inputDir[16:] + x for x in ls if x.endswith('root')]
    )
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string(ntupleFile)
)

process.p = cms.Path(process.simMuonGEMCSCPadDigis + process.GEMCSCAnalyzer)

