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
import os
process = cms.Process('AnalyzerPixelDigiMaps')

#################################################################################################
# global tag
#################################################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

#################################################################################################
# load the specific tracker geometry
#################################################################################################
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5D_cff')

#################################################################################################
# load the magnetic field
#################################################################################################
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
#process.load('Configuration.StandardSequences.MagneticField_40T_cff')
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

#################################################################################################
# define the source and maximum number of events to generate and simulate
#################################################################################################
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
#     fileNames = cms.untracked.vstring('file:TenMuPt_2_ExtendedPhase2TkBE5D_5000_DIGI_L1_DIGI2RAW_L1TT_RECO.root')
     fileNames = cms.untracked.vstring('file:TenMuPt_0_100_ExtendedPhase2TkBE5D_10000_DIGI_L1_DIGI2RAW_L1TT_RECO.root')
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

#################################################################################################
# load the analyzer
#################################################################################################
process.AnalyzerPixelDigiMaps = cms.EDAnalyzer("AnalyzerPixelDigiMaps",
#    DebugMode = cms.bool(True)
)

#################################################################################################
# define output file and message logger
#################################################################################################
process.TFileService = cms.Service("TFileService",
  fileName = cms.string('file:AnalyzerPixelDigiMaps_ExtendedPhase2TkBE5D.root')
)

process.eca = cms.EDAnalyzer("EventContentAnalyzer")
process.eca_step = cms.Path(process.eca)


#################################################################################################
# define the final path to be fed to cmsRun
#################################################################################################
process.p = cms.Path( process.AnalyzerPixelDigiMaps )

process.schedule = cms.Schedule(process.eca_step,process.p)

