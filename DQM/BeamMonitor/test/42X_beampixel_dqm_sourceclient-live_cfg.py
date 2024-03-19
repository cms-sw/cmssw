import FWCore.ParameterSet.Config as cms

process = cms.Process("BeamPixel")


#----------------------------
# Common Stuff for PP and H.I 
#----------------------------
process.load("DQM.Integration.test.inputsource_cfi")

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
process.dqmEnv.subSystemFolder = "BeamPixel"


#----------------------------
# Sub-system Configuration
#----------------------------
### @@@@@@ Comment when running locally @@@@@@ ###
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
### @@@@@@ Comment when running locally @@@@@@ ###
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi")
process.load("RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi")




#----------------------------
# Define common Sequence
#----------------------------
process.dqmmodules = cms.Sequence(process.dqmEnv 
                                + process.dqmSaver)


process.phystrigger = cms.Sequence(
                         process.hltTriggerTypeFilter
                         ### To use the L1 Filter uncomment the following line ###
                         #*process.gtDigis
                         #*process.hltLevel1GTSeed
       )




#----------------------------
# Proton-Proton Stuff
#----------------------------
if (process.runType.getRunType() == process.runType.pp_run or process.runType.getRunType() == process.runType.cosmic_run):
    print "Running pp paths"

    process.EventStreamHttpReader.consumerName = "Beam Pixel DQM Consumer"
    process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('HLT_L1*',
                                                                                              'HLT_Jet*',
                                                                                              'HLT_*Cosmic*',
                                                                                              'HLT_HT*',
                                                                                              'HLT_MinBias_*',
                                                                                              'HLT_Physics*',
                                                                                              'HLT_ZeroBias*'))
    process.load("Configuration.StandardSequences.Reconstruction_cff") ## pp reco sequence

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
    process.pixelVertexDQM = DQMEDAnalyzer('Vx3DHLTAnalyzer',
                                            vertexCollection = cms.InputTag("pixelVertices"),
                                            debugMode        = cms.bool(True),
                                            nLumiReset       = cms.uint32(1),
                                            dataFromFit      = cms.bool(True),
                                            minNentries      = cms.uint32(20),
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

    process.pixelVertices.TkFilterParameters.minPt = process.pixelTracks.RegionFactoryPSet.RegionPSet.ptMin

    process.reconstruction_step = cms.Sequence(
                                           process.siPixelDigis*
                                           process.offlineBeamSpot*
                                           process.siPixelClusters*
                                           process.siPixelRecHits*
                                           process.pixelTracks*
                                           process.pixelVertices*
                                           process.pixelVertexDQM)


    process.p = cms.Path(process.phystrigger 
                        *process.reconstruction_step 
                        *process.dqmmodules)






