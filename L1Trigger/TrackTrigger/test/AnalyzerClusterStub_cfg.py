####################################################
# SLHCUpgradeSimulations                           #
# Configuration file for Full Workflow             #
# Step 2 (again)                                   #
# Validate the L1 track trigger inputs             #
# TTCluster & TTStub.                              #
####################################################
# Author: Nicola Pozzobon                          #
# CERN, August 2012                                #
# Updated: Ian Tomalin                             #
# RAL, May 2026                                    #
####################################################

################################################################################
# import of general framework
################################################################################
import FWCore.ParameterSet.Config as cms

process = cms.Process('AnalyzerClusterStub')

GEOMETRY = "D110"

################################################################################
# global tag
################################################################################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

################################################################################
# load the specific tracker geometry
################################################################################
process.load('Configuration.Geometry.GeometryExtendedRun4' + GEOMETRY + 'Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4' + GEOMETRY +'_cff')

################################################################################
# load the magnetic field
################################################################################
process.load('Configuration.StandardSequences.MagneticField_cff')

################################################################################
# define the source and maximum number of events to generate and simulate
################################################################################
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('/store/relval/CMSSW_15_1_0_pre5/RelValDoubleMuFlatPt1p5To8/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_RV269_Run4D110_noPU-v1/2590000/1172421f-823f-420f-8ec9-3de20dd6dda4.root')
    #fileNames = cms.untracked.vstring('/store/relval/CMSSW_15_1_0_pre5/RelValTTbar_14TeV_TuneCP5/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_RV269_Run4D110_noPU-v1/2590000/02a7204c-3f1c-4858-9350-2f506d729bbc.root')
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_15_1_0_pre5/RelValTTbar_14TeV_TuneCP5/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_RV269_Run4D110_PU-v2/2590000/0f0bcfd3-dafe-4dda-8d39-9765f6eae68e.root')
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(10)
)

################################################################################
# load the analyzer
################################################################################
process.AnalyzerClusterStub = cms.EDAnalyzer("AnalyzerClusterStub",
  DebugMode = cms.bool(True),
  TTClusterInputTag = cms.InputTag("TTClustersFromPhase2TrackerDigis", "ClusterInclusive"),
  TTStubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),
  TTClusterAssocInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterInclusive"),
  TTStubAssocInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
  TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
  TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),   
)

#################################################################################################
# define output file and message logger
#################################################################################################
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('file:AnalyzerClusterStub.root'),
    closeFileFast = cms.untracked.bool(True)
)

#################################################################################################
# define the final path to be fed to cmsRun
#################################################################################################
process.p = cms.Path( process.AnalyzerClusterStub )



