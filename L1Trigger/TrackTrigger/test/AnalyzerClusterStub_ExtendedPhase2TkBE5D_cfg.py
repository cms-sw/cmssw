####################################################
# SLHCUpgradeSimulations                           #
# Configuration file for Full Workflow             #
# Step 2 (again)                                   #
# Understand if everything is fine with            #
# L1TkCluster e L1TkStub                           #
####################################################
# Nicola Pozzobon                                  #
# CERN, August 2012                                #
####################################################

#################################################################################################
# import of general framework
#################################################################################################
import FWCore.ParameterSet.Config as cms
#--------------import os
process = cms.Process('AnalyzerClusterStub')

#################################################################################################
# global tag
#################################################################################################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

#################################################################################################
# load the specific tracker geometry
#################################################################################################
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5D_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')

#################################################################################################
# load the magnetic field
#################################################################################################
process.load('Configuration.StandardSequences.MagneticField_cff')

#################################################################################################
# define the source and maximum number of events to generate and simulate
#################################################################################################
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
     fileNames = cms.untracked.vstring('file:TenMuPt_0_20_ExtendedPhase2TkBE5D_10000_DIGI_L1_DIGI2RAW_L1TT_RECO.root')
#     fileNames = cms.untracked.vstring('file:TenMuPt_0_20_ExtendedPhase2TkBE5D_10000_DIGI_L1_DIGI2RAW_L1TT_RECO_140PU.root')
#     fileNames = cms.untracked.vstring('file:test.root')

#     fileNames = cms.untracked.vstring('file:TenMuPt_2_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root',
#                                       #'file:TenMuPt_10_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root',
#                                       #'file:TenMuPt_100_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root'
#     )


#     fileNames = cms.untracked.vstring('file:TenPiPt_0_50_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root')
#     fileNames = cms.untracked.vstring('file:TenElePt_0_50_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root')
#     fileNames = cms.untracked.vstring('file:DYTauTau_ExtendedPhase2TkBE5D_750_DIGI_L1_DIGI2RAW_L1TT_RECO.root')
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

#################################################################################################
# load the analyzer
#################################################################################################
process.AnalyzerClusterStub = cms.EDAnalyzer("AnalyzerClusterStub",
    DebugMode = cms.bool(True)
)

#################################################################################################
# define output file and message logger
#################################################################################################
process.TFileService = cms.Service("TFileService",
#  fileName = cms.string('file:AnalyzerClusterStub_ExtendedPhase2TkBE5D_MuonPU140.root')
#  fileName = cms.string('file:AnalyzerClusterStub_ExtendedPhase2TkBE5D_Pion.root')
  fileName = cms.string('file:AnalyzerClusterStub_ExtendedPhase2TkBE5D_Muon.root')
#  fileName = cms.string('file:AnalyzerClusterStub_ExtendedPhase2TkBE5D_DYTauTau.root')
)

#################################################################################################
# define the final path to be fed to cmsRun
#################################################################################################
process.p = cms.Path( process.AnalyzerClusterStub )



