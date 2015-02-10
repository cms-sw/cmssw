import FWCore.ParameterSet.Config as cms

process = cms.Process("BeamPixel")


#----------------------------
# Event Source
#----------------------------
### @@@@@@ Comment when running locally @@@@@@ ###
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = "Beam Pixel DQM Consumer"
process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('HLT_L1*','HLT_TrackerCosmics','HLT_Jet*'))
### @@@@@@ Comment when running locally @@@@@@ ###


#----------------------------
# Filters
#----------------------------
# HLT Filter
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1
process.physicsBitSelector = cms.EDFilter("PhysDecl",
                                          applyfilter = cms.untracked.bool(True),
                                          debugOn     = cms.untracked.bool(False))
# L1 Filter
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff")
process.load("HLTrigger.HLTfilters.hltLevel1GTSeed_cfi")
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string("0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND (NOT 42 OR 43) AND (42 OR NOT 43)")


#----------------------------
# DQM Environment
#----------------------------
process.load("DQM.Integration.test.environment_cfi")
### @@@@@@ Un-comment when running locally @@@@@@ ###
#process.DQM.collectorHost = ''
### @@@@@@ Un-comment when running locally @@@@@@ ###
process.dqmEnv.subSystemFolder = "BeamPixel"


#----------------------------
# Sub-system Configuration
#----------------------------
### @@@@@@ Comment when running locally @@@@@@ ###
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
###########################################
### TEMPORARY: using offline alignments ###
###########################################
process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_31X_GLOBALTAG"
process.GlobalTag.globaltag = "GR10_E_V6::All"
process.GlobalTag.pfnPrefix=cms.untracked.string("frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/")
### @@@@@@ Comment when running locally @@@@@@ ###
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi")
process.load("RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi")


### @@@@@@ Un-comment when running locally @@@@@@ ###
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# RECO data taking february 18th 2010
#process.GlobalTag.globaltag = "GR09_R_35X_V2::All"
###### Which data ######
#process.load("DataDec09_RecoMinBias_Feb18th_Skim_Run124120_cff")
#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
###### DQM Saver ######
#process.dqmSaver.dirName           = cms.untracked.string("/nfshome0/yumiceva/BeamMonitorDQM/")
#process.dqmSaver.saveAtJobEnd      = cms.untracked.bool(False)
#process.dqmSaver.saveByTime        = cms.untracked.int32(-1)
#process.dqmSaver.saveByRun         = cms.untracked.int32(-1)
#process.dqmSaver.forceRunNumber    = cms.untracked.int32(-1)
#process.dqmSaver.saveByLumiSection = cms.untracked.int32(-1)
#process.dqmSaver.saveByMinute      = cms.untracked.int32(-1)
###### Output file ######
#process.Output = cms.OutputModule("PoolOutputModule",
#                                  fileName = cms.untracked.string( "/tmp/dinardo/BeamPixelResults.root" ),
#                                  outputCommands = cms.untracked.vstring( "drop *",
#                                                                          "keep *_*_*_BeamPixel"))
### @@@@@@ Un-comment when running locally @@@@@@ ###


#----------------------------
# pixelVertexDQM Configuration
#----------------------------
process.pixelVertexDQM = cms.EDAnalyzer("Vx3DHLTAnalyzer",
                                        vertexCollection = cms.InputTag("pixelVertices"),
                                        debugMode        = cms.bool(True),
                                        nLumiReset       = cms.uint32(1),
                                        dataFromFit      = cms.bool(True),
                                        minNentries      = cms.uint32(35),
                                        # If the histogram has at least "minNentries" then extract Mean and RMS,
                                        # or, if we are performing the fit, the number of vertices must be greater
                                        # than minNentries otherwise it waits for other nLumiReset
                                        xRange           = cms.double(2.0),
                                        xStep            = cms.double(0.001),
                                        yRange           = cms.double(2.0),
                                        yStep            = cms.double(0.001),
                                        zRange           = cms.double(30.0),
                                        zStep            = cms.double(0.05),
                                        VxErrCorr        = cms.double(1.5),
                                        fileName         = cms.string("/nfshome0/yumiceva/BeamMonitorDQM/BeamPixelResults.txt"))
if process.dqmSaver.producer.value() is "Playback":
  process.pixelVertexDQM.fileName = cms.string("/nfshome0/dqmdev/BeamMonitorDQM/BeamPixelResults.txt") 
else:
  process.pixelVertexDQM.fileName = cms.string("/nfshome0/dqmpro/BeamMonitorDQM/BeamPixelResults.txt")


#----------------------------
# Pixel-Tracks Configuration
#----------------------------
process.PixelTrackReconstructionBlock.RegionFactoryPSet.ComponentName = "GlobalRegionProducer"


#----------------------------
# Pixel-Vertices Configuration
#----------------------------
process.pixelVertices.useBeamConstraint = False
process.pixelVertices.TkFilterParameters.minPt = process.pixelTracks.RegionFactoryPSet.RegionPSet.ptMin


#----------------------------
# Define Sequence
#----------------------------
process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)

process.phystrigger = cms.Sequence(
  process.hltTriggerTypeFilter
#  process.gtDigis*
#  process.hltLevel1GTSeed
  )

process.reconstruction_step = cms.Sequence(
    process.siPixelDigis*
    process.offlineBeamSpot*
    process.siPixelClusters*
    process.siPixelRecHits*
    process.pixelTracks*
    process.pixelVertices*
    process.pixelVertexDQM)


#----------------------------
# Define Path
#----------------------------
process.p = cms.Path(process.phystrigger * process.reconstruction_step * process.dqmmodules)
