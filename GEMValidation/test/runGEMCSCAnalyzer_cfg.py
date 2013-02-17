import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
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
process.GEMCSCAnalyzer.ntupleTrackChamberDelta = False
process.GEMCSCAnalyzer.ntupleTrackEff = True
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
import os

inputDir = dirPt20 ; ntupleFile = 'gem_csc_delta_pt20_or16.root'
inputDir = dirPt5  ; ntupleFile = 'gem_csc_delta_pt5_or16.root'

#inputDir = dirPt20 ; ntupleFile = 'gem_csc_delta_pt20_or8.root'
#inputDir = dirPt5  ; ntupleFile = 'gem_csc_delta_pt5_or8.root'

#inputDir = dirPt20 ; ntupleFile = 'gem_csc_delta_pt20_or4_lct.root'
#inputDir = dirPt5  ; ntupleFile = 'gem_csc_delta_pt5_or4_lct.root'

#inputDir = dirPt5Pt40  ; ntupleFile = 'gem_csc_delta_pt5pt40_or4.root'
inputDir = dirPt5Pt40  ; ntupleFile = 'gem_csc_eff_pt5pt40_or4.root'

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

