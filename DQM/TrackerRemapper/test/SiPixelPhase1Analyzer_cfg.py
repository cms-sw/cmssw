import FWCore.ParameterSet.Config as cms 

# Standard includes
process = cms.Process("SiPixelPhase1Analyzer")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

# MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.SiPixelPhase1Analyzer=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiPixelPhase1Analyzer = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )

# Source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
      '/store/relval/CMSSW_11_2_0_pre6/RelValQCD_FlatPt_15_3000HS_14/GEN-SIM-RECO/112X_mcRun3_2021_realistic_v7-v1/20000/47B75EDE-AAB4-C944-802B-15B1EB4BA29A.root'
  )
) 

# Output root file name:
process.TFileService = cms.Service("TFileService", fileName = cms.string('pixelMaps.root') )

MODE_ANALYZE = 0
MODE_REMAP = 1

process.demo = cms.EDAnalyzer('SiPixelPhase1Analyzer',
                              opMode = cms.untracked.uint32(MODE_ANALYZE),
                              src = cms.InputTag("generalTracks"),
                              debugFileName = cms.untracked.string("debug.txt"),                              
                              remapRootFileName = cms.untracked.vstring("dqmFile.root"), #only one input is allowed now
                              isBarrelSource = cms.untracked.vuint32(0, 0, 1),
                              pathToHistograms = cms.untracked.vstring(
                                  "DQMData/Run 1/PixelPhase1/Run summary/Phase1_MechanicalView/PXForward/",
                                  "DQMData/Run 1/PixelPhase1/Run summary/Phase1_MechanicalView/PXForward/",
                                  "DQMData/Run 1/PixelPhase1/Run summary/Phase1_MechanicalView/PXBarrel/"),
                              baseHistogramName = cms.untracked.vstring(
                                  "num_clusters_per_PXDisk_per_SignedBladePanel_PXRing",
                                  "num_digis_per_PXDisk_per_SignedBladePanel_PXRing",
                                  "num_digis_per_SignedModule_per_SignedLadder_PXLayer")
                              )

process.p = cms.Path(process.demo)
