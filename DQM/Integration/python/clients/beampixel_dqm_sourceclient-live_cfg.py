from __future__ import print_function
import FWCore.ParameterSet.Config as cms

import sys
if 'runkey=hi_run' in sys.argv:
  from Configuration.Eras.Era_Run3_pp_on_PbPb_approxSiStripClusters_cff import Run3_pp_on_PbPb_approxSiStripClusters
  process = cms.Process("BeamPixel", Run3_pp_on_PbPb_approxSiStripClusters)
else:
  from Configuration.Eras.Era_Run3_cff import Run3
  process = cms.Process("BeamPixel", Run3)

unitTest = False
if 'unitTest=True' in sys.argv:
    unitTest = True


#----------------------------
# Common for PP and HI running
#----------------------------
if unitTest == True:
    process.load("DQM.Integration.config.unittestinputsource_cfi")
    from DQM.Integration.config.unittestinputsource_cfi import options
else:
    process.load("DQM.Integration.config.inputsource_cfi")
    from DQM.Integration.config.inputsource_cfi import options
# Use this to run locally (for testing purposes)
#process.load("DQM.Integration.config.fileinputsource_cfi")
#from DQM.Integration.config.fileinputsource_cfi import options


#----------------------------
# HLT Filter
#----------------------------
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter", SelectedTriggerType = cms.int32(1))


#----------------------------
# DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "BeamPixel"
process.dqmSaver.tag = "BeamPixel"
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'BeamPixel'
process.dqmSaverPB.runNumber = options.runNumber

#----------------------------
# Conditions
#----------------------------
# Use this to run locally (for testing purposes), choose the right GT
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, "auto:run3_data", "")
# Otherwise use this
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")


#----------------------------
# Sub-system Configuration
#----------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")


#----------------------------
# Define Sequences
#----------------------------
process.dqmModules  = cms.Sequence(process.dqmEnv + process.dqmSaver + process.dqmSaverPB)
process.physTrigger = cms.Sequence(process.hltTriggerTypeFilter)


#----------------------------
# Process Customizations
#----------------------------
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer


#----------------------------
# Tracking General Configuration
#----------------------------
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")


#----------------------------
# Pixel-Tracks&Vertices Config
#----------------------------
from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
process.siPixelClusterShapeCachePreSplitting = siPixelClusterShapeCache.clone(src = 'siPixelClustersPreSplitting')
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi")
process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
from RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi import *
process.pixelVertices = pixelVertices.clone(
    TkFilterParameters = dict(
	minPt = process.pixelTracksTrackingRegions.RegionPSet.ptMin)
)
process.pixelTracksTrackingRegions.RegionPSet.originRadius     = 0.4
process.pixelTracksTrackingRegions.RegionPSet.originHalfLength = 15.
process.pixelTracksTrackingRegions.RegionPSet.originXPos       = 0.08
process.pixelTracksTrackingRegions.RegionPSet.originYPos       = -0.03
process.pixelTracksTrackingRegions.RegionPSet.originZPos       = 0.


#----------------------------
# Proton-Proton Specific Section
#----------------------------
if (process.runType.getRunType() == process.runType.pp_run or process.runType.getRunType() == process.runType.pp_run_stage1 or 
    process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1 or 
    process.runType.getRunType() == process.runType.hpu_run or process.runType.getRunType() == process.runType.commissioning_run ):
    print("[beampixel_dqm_sourceclient-live_cfg]::running pp")


    #----------------------------
    # Tracking Configuration
    #----------------------------
    process.castorDigis.InputLabel           = "rawDataCollector"
    process.csctfDigis.producer              = "rawDataCollector"
    process.dttfDigis.DTTF_FED_Source        = "rawDataCollector"
    process.ecalDigis.cpu.InputLabel         = "rawDataCollector"
    process.ecalPreshowerDigis.sourceTag     = "rawDataCollector"
    process.gctDigis.inputLabel              = "rawDataCollector"
    process.gtDigis.DaqGtInputTag            = "rawDataCollector"
    process.hcalDigis.InputLabel             = "rawDataCollector"
    process.muonCSCDigis.InputObjects        = "rawDataCollector"
    process.muonDTDigis.inputLabel           = "rawDataCollector"
    process.muonRPCDigis.InputLabel          = "rawDataCollector"
    process.scalersRawToDigi.scalersInputTag = "rawDataCollector"
    process.siPixelDigis.cpu.InputLabel      = "rawDataCollector"
    process.siStripDigis.ProductLabel        = "rawDataCollector"

    
    #----------------------------
    # pixelVertexDQM Config
    #----------------------------
    process.pixelVertexDQM = DQMEDAnalyzer('Vx3DHLTAnalyzer',
                                            vertexCollection   = cms.untracked.InputTag("pixelVertices"),
                                            pixelHitCollection = cms.untracked.InputTag("siPixelRecHitsPreSplitting"),
                                            debugMode          = cms.bool(True),
                                            nLumiFit           = cms.uint32(2),
                                            maxLumiIntegration = cms.uint32(15),
                                            nLumiXaxisRange    = cms.uint32(5000),
                                            dataFromFit        = cms.bool(True),
                                            minNentries        = cms.uint32(20),
                                            # If the histogram has at least "minNentries" then extract Mean and RMS,
                                            # or, if we are performing the fit, the number of vertices must be greater
                                            # than minNentries otherwise it waits for other nLumiFit
                                            xRange             = cms.double(0.8),
                                            xStep              = cms.double(0.001),
                                            yRange             = cms.double(0.8),
                                            yStep              = cms.double(0.001),
                                            zRange             = cms.double(30.0),
                                            zStep              = cms.double(0.04),
                                            VxErrCorr          = cms.double(1.0), # Was 1.2, changed to 1.0 in Run3 13.6 TeV collisions - Keep checking this with later release
                                            minVxDoF           = cms.double(10.0),
                                            minVxWgt           = cms.double(0.5),
                                            fileName           = cms.string("/nfshome0/dqmdev/BeamMonitorDQM/BeamPixelResults.txt"))


#----------------------------
# Heavy Ion Specific Section
#----------------------------
if (process.runType.getRunType() == process.runType.hi_run):
    print("[beampixel_dqm_sourceclient-live_cfg]::running HI")


    #----------------------------
    # Tracking Configuration
    #----------------------------
    process.castorDigis.InputLabel           = "rawDataRepacker"
    process.csctfDigis.producer              = "rawDataRepacker"
    process.dttfDigis.DTTF_FED_Source        = "rawDataRepacker"
    process.ecalDigis.cpu.InputLabel         = "rawDataRepacker"
    process.ecalPreshowerDigis.sourceTag     = "rawDataRepacker"
    process.gctDigis.inputLabel              = "rawDataRepacker"
    process.gtDigis.DaqGtInputTag            = "rawDataRepacker"
    process.hcalDigis.InputLabel             = "rawDataRepacker"
    process.muonCSCDigis.InputObjects        = "rawDataRepacker"
    process.muonDTDigis.inputLabel           = "rawDataRepacker"
    process.muonRPCDigis.InputLabel          = "rawDataRepacker"
    process.scalersRawToDigi.scalersInputTag = "rawDataRepacker"
    process.siPixelDigis.cpu.InputLabel      = "rawDataRepacker"
    process.siStripDigis.ProductLabel        = "rawDataRepacker"


    #----------------------------
    # pixelVertexDQM Config
    #----------------------------
    process.pixelVertexDQM = DQMEDAnalyzer('Vx3DHLTAnalyzer',
                                            vertexCollection   = cms.untracked.InputTag("pixelVertices"),
                                            pixelHitCollection = cms.untracked.InputTag("siPixelRecHitsPreSplitting"),
                                            debugMode          = cms.bool(True),
                                            nLumiFit           = cms.uint32(5),
                                            maxLumiIntegration = cms.uint32(15),
                                            nLumiXaxisRange    = cms.uint32(5000),
                                            dataFromFit        = cms.bool(True),
                                            minNentries        = cms.uint32(20),
                                            # If the histogram has at least "minNentries" then extract Mean and RMS,
                                            # or, if we are performing the fit, the number of vertices must be greater
                                            # than minNentries otherwise it waits for other nLumiFit
                                            xRange             = cms.double(0.8),
                                            xStep              = cms.double(0.001),
                                            yRange             = cms.double(0.8),
                                            yStep              = cms.double(0.001),
                                            zRange             = cms.double(30.0),
                                            zStep              = cms.double(0.04),
                                            VxErrCorr          = cms.double(1.0), # Was 1.2, changed to 1.0 in Run3 13.6 TeV collisions - Keep checking this with later release
                                            minVxDoF           = cms.double(10.0),
                                            minVxWgt           = cms.double(0.5),
                                            fileName           = cms.string("/nfshome0/dqmdev/BeamMonitorDQM/BeamPixelResults.txt"))


#----------------------------
# File to save beamspot info
#----------------------------
if process.dqmRunConfig.type.value() == "production":
    process.pixelVertexDQM.fileName = "/nfshome0/dqmpro/BeamMonitorDQM/BeamPixelResults.txt"
else:
    process.pixelVertexDQM.fileName = "/nfshome0/dqmdev/BeamMonitorDQM/BeamPixelResults.txt"
print("[beampixel_dqm_sourceclient-live_cfg]::saving DIP file into " + str(process.pixelVertexDQM.fileName))


#----------------------------
# Pixel-Tracks&Vertices Reco
#----------------------------
process.reconstructionStep = cms.Sequence(process.siPixelDigis*
                                          process.siStripDigis*
                                          process.striptrackerlocalreco*
                                          process.offlineBeamSpot*
                                          process.siPixelClustersPreSplitting*
                                          process.siPixelRecHitsPreSplitting*
                                          process.siPixelClusterShapeCachePreSplitting*
                                          process.recopixelvertexing)


#----------------------------
# Define Path
#----------------------------
process.p = cms.Path(process.scalersRawToDigi*process.physTrigger*process.reconstructionStep*process.pixelVertexDQM*process.dqmModules)
