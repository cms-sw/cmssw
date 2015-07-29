import FWCore.ParameterSet.Config as cms

process = cms.Process("BeamPixel")


#----------------------------
# Common for PP and HI running
#----------------------------
### @@@@@@ Comment when running locally @@@@@@ ###
process.load("DQM.Integration.config.inputsource_cfi")

#----------------------------
# HLT Filter
#----------------------------
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",SelectedTriggerType = cms.int32(1))


#----------------------------
# DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "BeamPixel"
process.dqmSaver.tag = "BeamPixel"


#----------------------------
# Sub-system Configuration
#----------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")


#----------------------------
# Define Sequences
#----------------------------
process.dqmmodules  = cms.Sequence(process.dqmEnv + process.dqmSaver)
process.phystrigger = cms.Sequence(process.hltTriggerTypeFilter)


#----------------------------
# Process Customizations
#----------------------------
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)


#----------------------------
# Proton-Proton Specific Part
#----------------------------
if (process.runType.getRunType() == process.runType.pp_run or process.runType.getRunType() == process.runType.pp_run_stage1 or 
    process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1 or 
    process.runType.getRunType() == process.runType.hpu_run):
    print "[beampixel_dqm_sourceclient-live_cfg]::running pp"

    process.castorDigis.InputLabel           = cms.InputTag("rawDataCollector")
    process.csctfDigis.producer              = cms.InputTag("rawDataCollector")
    process.dttfDigis.DTTF_FED_Source        = cms.InputTag("rawDataCollector")
    process.ecalDigis.InputLabel             = cms.InputTag("rawDataCollector")
    process.ecalPreshowerDigis.sourceTag     = cms.InputTag("rawDataCollector")
    process.gctDigis.inputLabel              = cms.InputTag("rawDataCollector")
    process.gtDigis.DaqGtInputTag            = cms.InputTag("rawDataCollector")
    process.gtEvmDigis.EvmGtInputTag         = cms.InputTag("rawDataCollector")
    process.hcalDigis.InputLabel             = cms.InputTag("rawDataCollector")
    process.muonCSCDigis.InputObjects        = cms.InputTag("rawDataCollector")
    process.muonDTDigis.inputLabel           = cms.InputTag("rawDataCollector")
    process.muonRPCDigis.InputLabel          = cms.InputTag("rawDataCollector")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
    process.siPixelDigis.InputLabel          = cms.InputTag("rawDataCollector")
    process.siStripDigis.ProductLabel        = cms.InputTag("rawDataCollector")

    process.load("Configuration.StandardSequences.Reconstruction_cff")


    #----------------------------
    # pixelVertexDQM Configuration
    #----------------------------
    process.pixelVertexDQM = cms.EDAnalyzer("Vx3DHLTAnalyzer",
                                            vertexCollection   = cms.untracked.InputTag("pixelVertices"),
                                            #pixelHitCollection = cms.untracked.InputTag("siPixelRecHits"),
                                            pixelHitCollection = cms.untracked.InputTag("siPixelRecHitsPreSplitting"),
                                            debugMode          = cms.bool(True),
                                            nLumiReset         = cms.uint32(2),
                                            dataFromFit        = cms.bool(True),
                                            minNentries        = cms.uint32(20),
                                            # If the histogram has at least "minNentries" then extract Mean and RMS,
                                            # or, if we are performing the fit, the number of vertices must be greater
                                            # than minNentries otherwise it waits for other nLumiReset
                                            xRange             = cms.double(1.0),
                                            xStep              = cms.double(0.001),
                                            yRange             = cms.double(1.0),
                                            yStep              = cms.double(0.001),
                                            zRange             = cms.double(30.0),
                                            zStep              = cms.double(0.05),
                                            VxErrCorr          = cms.double(1.3),  # Keep checking this with later release
                                            fileName           = cms.string("/nfshome0/yumiceva/BeamMonitorDQM/BeamPixelResults.txt"))
    if process.dqmRunConfig.type.value() is "playback":
       process.pixelVertexDQM.fileName = cms.string("/nfshome0/dqmdev/BeamMonitorDQM/BeamPixelResults.txt")
    else:
       process.pixelVertexDQM.fileName = cms.string("/nfshome0/dqmpro/BeamMonitorDQM/BeamPixelResults.txt")




    process.load("RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi")
    #pixel  track/vertices reco
    process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
    process.pixelVertices.TkFilterParameters.minPt = process.pixelTracks.RegionFactoryPSet.RegionPSet.ptMin
    process.offlinePrimaryVertices.TrackLabel = cms.InputTag("pixelTracks")
    #process.dqmBeamMonitor.PVFitter.errorScale = 1.25 #keep checking this with new release expected close to 1.2
    
    from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
    process.PixelLayerTriplets.BPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
    process.PixelLayerTriplets.FPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
    from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
    process.pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.clusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCachePreSplitting')

    #----------------------------
    # Pixel-Vertices Configuration
    #----------------------------
    process.reconstruction_step  = cms.Sequence(process.siPixelDigis*
                                               process.offlineBeamSpot*
                                               process.siPixelClustersPreSplitting*
                                               process.siPixelRecHitsPreSplitting*
                                               process.siPixelClusterShapeCachePreSplitting*
                                               process.recopixelvertexing
                                               )


    #----------------------------
    # Define Path
    #----------------------------
    process.p = cms.Path(process.phystrigger*process.reconstruction_step*process.pixelVertexDQM*process.dqmmodules)




#----------------------------
# Heavy Ion Specific Part
#----------------------------
if (process.runType.getRunType() == process.runType.hi_run):
    print "[beampixel_dqm_sourceclient-live_cfg]::running HI"

    process.castorDigis.InputLabel           = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer              = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source        = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel             = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag     = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel              = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag            = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag         = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel             = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects        = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel           = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel          = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel          = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel        = cms.InputTag("rawDataRepacker")

    process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff")


    #----------------------------
    # pixelVertexDQM Configuration
    #----------------------------
    process.pixelVertexDQM = cms.EDAnalyzer("Vx3DHLTAnalyzer",
                                            vertexCollection   = cms.untracked.InputTag("hiSelectedVertex"),
                                            pixelHitCollection = cms.untracked.InputTag("siPixelRecHits"),
                                            debugMode          = cms.bool(True),
                                            nLumiReset         = cms.uint32(5),
                                            dataFromFit        = cms.bool(True),
                                            minNentries        = cms.uint32(20),
                                            # If the histogram has at least "minNentries" then extract Mean and RMS,
                                            # or, if we are performing the fit, the number of vertices must be greater
                                            # than minNentries otherwise it waits for other nLumiReset
                                            xRange             = cms.double(1.0),
                                            xStep              = cms.double(0.001),
                                            yRange             = cms.double(1.0),
                                            yStep              = cms.double(0.001),
                                            zRange             = cms.double(30.0),
                                            zStep              = cms.double(0.05),
                                            VxErrCorr          = cms.double(1.3),  # Keep checking this with later release
                                            fileName           = cms.string("/nfshome0/yumiceva/BeamMonitorDQM/BeamPixelResults.txt"))
    if process.dqmRunConfig.type.value() is "playback":
       process.pixelVertexDQM.fileName = cms.string("/nfshome0/dqmdev/BeamMonitorDQM/BeamPixelResults.txt")
    else:
       process.pixelVertexDQM.fileName = cms.string("/nfshome0/dqmpro/BeamMonitorDQM/BeamPixelResults.txt")


    #----------------------------
    # Pixel-Vertices Configuration
    #----------------------------
    process.reconstruction_step = cms.Sequence(process.siPixelDigis*
                                               process.offlineBeamSpot*
                                               process.siPixelClusters*
                                               process.siPixelRecHits*
                                               process.hiPixelVertices*
                                               process.siPixelClusterShapeCache*
                                               process.PixelLayerTriplets*
                                               process.pixelTracks*
                                               process.pixelVertices)


    #----------------------------
    # Define Path
    #----------------------------
    process.p = cms.Path(process.phystrigger*process.reconstruction_step*process.pixelVertexDQM*process.dqmmodules)
    
