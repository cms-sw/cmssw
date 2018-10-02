from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("BeamPixel", eras.Run2_2018)


#----------------------------
# Common for PP and HI running
#----------------------------
# Use this to run locally (for testing purposes)
#process.load("DQM.Integration.config.fileinputsource_cfi")
# Otherwise use this
process.load("DQM.Integration.config.inputsource_cfi")


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
process.dqmModules  = cms.Sequence(process.dqmEnv + process.dqmSaver)
process.physTrigger = cms.Sequence(process.hltTriggerTypeFilter)


#----------------------------
# Process Customizations
#----------------------------
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer


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
process.siPixelDigis.InputLabel          = cms.InputTag("rawDataRepacker")
process.siStripDigis.ProductLabel        = cms.InputTag("rawDataRepacker")

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
    process.runType.getRunType() == process.runType.hpu_run):
    print("[beampixel_dqm_sourceclient-live_cfg]::running pp")


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
    # Pixel-Tracks&Vertices Config
    #----------------------------
    from RecoVertex.PrimaryVertexProducer.TkClusParameters_cff import DA_vectParameters
    offlinePrimaryVertices = cms.EDProducer(
        "PrimaryVertexProducer",
        verbose       = cms.untracked.bool(False),
        TrackLabel    = cms.InputTag("generalTracks"),
        beamSpotLabel = cms.InputTag("offlineBeamSpot"),
        TkFilterParameters           = cms.PSet(
            algorithm                = cms.string('filter'),
            maxNormalizedChi2        = cms.double(10.0),
            minPixelLayersWithHits   = cms.int32(2),
            minSiliconLayersWithHits = cms.int32(5),
            maxD0Significance        = cms.double(4.0), 
            minPt                    = cms.double(0.0),
            maxEta                   = cms.double(2.4),
            trackQuality             = cms.string("any")),
        TkClusParameters  = DA_vectParameters,
        vertexCollections = cms.VPSet(
            [cms.PSet(label             = cms.string(""),
                      algorithm         = cms.string("AdaptiveVertexFitter"),
                      chi2cutoff        = cms.double(2.5),
                      minNdof           = cms.double(0.0),
                      useBeamConstraint = cms.bool(False),
                      maxDistanceToBeam = cms.double(1.0)),
             cms.PSet(label             = cms.string("WithBS"),
                      algorithm         = cms.string('AdaptiveVertexFitter'),
                      chi2cutoff        = cms.double(2.5),
                      minNdof           = cms.double(2.0),
                      useBeamConstraint = cms.bool(True),
                      maxDistanceToBeam = cms.double(1.0))]))


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
