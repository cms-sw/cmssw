from __future__ import print_function
import FWCore.ParameterSet.Config as cms

import sys
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process("BeamPixel", Run2_2018)

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
#process.GlobalTag = GlobalTag(process.GlobalTag, "auto:run2_data", "")
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
process.pixelVertices.TkFilterParameters.minPt = process.pixelTracksTrackingRegions.RegionPSet.ptMin
process.pixelTracksTrackingRegions.RegionPSet.originRadius     = cms.double(0.4)
process.pixelTracksTrackingRegions.RegionPSet.originHalfLength = cms.double(15.)
process.pixelTracksTrackingRegions.RegionPSet.originXPos       = cms.double(0.08)
process.pixelTracksTrackingRegions.RegionPSet.originYPos       = cms.double(-0.03)
process.pixelTracksTrackingRegions.RegionPSet.originZPos       = cms.double(0.)


#----------------------------
# Proton-Proton Specific Section
#----------------------------
if (process.runType.getRunType() == process.runType.pp_run or process.runType.getRunType() == process.runType.pp_run_stage1 or 
    process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1 or 
    process.runType.getRunType() == process.runType.hpu_run):
    print("[beampixel_dqm_sourceclient-live_cfg]::running pp")


    #----------------------------
    # Tracking Configuration
    #----------------------------
    process.castorDigis.InputLabel           = cms.InputTag("rawDataCollector")
    process.csctfDigis.producer              = cms.InputTag("rawDataCollector")
    process.dttfDigis.DTTF_FED_Source        = cms.InputTag("rawDataCollector")
    process.ecalDigis.InputLabel             = cms.InputTag("rawDataCollector")
    process.ecalPreshowerDigis.sourceTag     = cms.InputTag("rawDataCollector")
    process.gctDigis.inputLabel              = cms.InputTag("rawDataCollector")
    process.gtDigis.DaqGtInputTag            = cms.InputTag("rawDataCollector")
    process.hcalDigis.InputLabel             = cms.InputTag("rawDataCollector")
    process.muonCSCDigis.InputObjects        = cms.InputTag("rawDataCollector")
    process.muonDTDigis.inputLabel           = cms.InputTag("rawDataCollector")
    process.muonRPCDigis.InputLabel          = cms.InputTag("rawDataCollector")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
    process.siPixelDigis.cpu.InputLabel      = cms.InputTag("rawDataCollector")
    process.siStripDigis.ProductLabel        = cms.InputTag("rawDataCollector")

    
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
                                            VxErrCorr          = cms.double(1.2), # Keep checking this with later release
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
    process.castorDigis.InputLabel           = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer              = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source        = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel             = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag     = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel              = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag            = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel             = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects        = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel           = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel          = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.cpu.InputLabel      = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel        = cms.InputTag("rawDataRepacker")


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
                                            VxErrCorr          = cms.double(1.2), # Keep checking this with later release
                                            minVxDoF           = cms.double(10.0),
                                            minVxWgt           = cms.double(0.5),
                                            fileName           = cms.string("/nfshome0/dqmdev/BeamMonitorDQM/BeamPixelResults.txt"))


#----------------------------
# File to save beamspot info
#----------------------------
if process.dqmRunConfig.type.value() is "production":
    process.pixelVertexDQM.fileName = cms.string("/nfshome0/dqmpro/BeamMonitorDQM/BeamPixelResults.txt")
else:
    process.pixelVertexDQM.fileName = cms.string("/nfshome0/dqmdev/BeamMonitorDQM/BeamPixelResults.txt")
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
