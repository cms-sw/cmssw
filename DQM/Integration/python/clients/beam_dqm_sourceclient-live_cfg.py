import FWCore.ParameterSet.Config as cms

process = cms.Process("BeamMonitor")

#----------------------------------------------                                                                                                                                    
# Switch to change between firstStep and Pixel
#-----------------------------------------------

runFirstStepTrk = True

#----------------------------
# Common part for PP and H.I Running
#-----------------------------
# for live online DQM in P5
process.load("DQM.Integration.config.inputsource_cfi")

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")

#--------------------------
# HLT Filter
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1)
)

#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.tag = 'BeamMonitor'

process.dqmEnvPixelLess = process.dqmEnv.clone()
process.dqmEnvPixelLess.subSystemFolder = 'BeamMonitor_PixelLess'

#----------------------------
# BeamMonitor
#-----------------------------

if(runFirstStepTrk):
  process.load("DQM.BeamMonitor.BeamMonitor_cff") # for reduced/firsStep/normal tracking
else:
  process.load("DQM.BeamMonitor.BeamMonitor_Pixel_cff") #for pixel tracks/vertices

process.load("DQM.BeamMonitor.BeamSpotProblemMonitor_cff")
#process.load("DQM.BeamMonitor.BeamMonitorBx_cff")
#process.load("DQM.BeamMonitor.BeamMonitor_PixelLess_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")


####  SETUP TRACKING RECONSTRUCTION ####
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

#---------------
# Calibration
#---------------
# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus
#process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi") 

# Change Beam Monitor variables
if process.dqmRunConfig.type.value() is "playback":
  process.dqmBeamMonitor.BeamFitter.WriteAscii = False
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmdev/BeamMonitorDQM/BeamFitResults.txt'
else:
  process.dqmBeamMonitor.BeamFitter.WriteAscii = True
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults.txt'
  #process.dqmBeamMonitor.BeamFitter.SaveFitResults = False
  #process.dqmBeamMonitor.BeamFitter.OutputFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.root'
  #process.dqmBeamMonitorBx.BeamFitter.WriteAscii = False
  #process.dqmBeamMonitorBx.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults_Bx.txt'


## TKStatus
process.dqmTKStatus = cms.EDAnalyzer("TKStatus",
        BeamFitter = cms.PSet(
        DIPFileName = process.dqmBeamMonitor.BeamFitter.DIPFileName
        )
)


process.dqmcommon = cms.Sequence(process.dqmEnv
                                *process.dqmSaver)

process.monitor = cms.Sequence(process.dqmBeamMonitor)


#------------------------------------------------------------
# BeamSpotProblemMonitor Modules
#-----------------------------------------------------------
process.dqmBeamSpotProblemMonitor.monitorName       = "BeamMonitor/BeamSpotProblemMonitor"
process.dqmBeamSpotProblemMonitor.AlarmONThreshold  = 10
process.dqmBeamSpotProblemMonitor.AlarmOFFThreshold = 12
process.dqmBeamSpotProblemMonitor.nCosmicTrk        = 10
process.dqmBeamSpotProblemMonitor.doTest            = False
if (runFirstStepTrk):
    process.dqmBeamSpotProblemMonitor.pixelTracks   = 'initialStepTracksPreSplitting'
else:
    process.dqmBeamSpotProblemMonitor.pixelTracks   = 'pixelTracks'


process.qTester = cms.EDAnalyzer("QualityTester",
                                 qtList = cms.untracked.FileInPath('DQM/BeamMonitor/test/BeamSpotAvailableTest.xml'),
                                 prescaleFactor = cms.untracked.int32(1),                               
                                 qtestOnEndLumi = cms.untracked.bool(True),
                                 testInEventloop = cms.untracked.bool(False),
                                 verboseQT =  cms.untracked.bool(True)                 
                                )

process.BeamSpotProblemModule = cms.Sequence( process.qTester
 	  	                             *process.dqmBeamSpotProblemMonitor
                                            )

#make it off for cosmic run
if ( process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1):
    process.dqmBeamSpotProblemMonitor.AlarmOFFThreshold = 5       #Should be < AlalrmONThreshold 
#-----------------------------------------------------------

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)


#--------------------------
# Proton-Proton Stuff
#--------------------------

if (process.runType.getRunType() == process.runType.pp_run or process.runType.getRunType() == process.runType.pp_run_stage1 or
    process.runType.getRunType() == process.runType.cosmic_run or process.runType.getRunType() == process.runType.cosmic_run_stage1 or 
    process.runType.getRunType() == process.runType.hpu_run):

    print "[beam_dqm_sourceclient-live_cfg]:: Running pp"

    process.castorDigis.InputLabel = cms.InputTag("rawDataCollector")
    process.csctfDigis.producer = cms.InputTag("rawDataCollector")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataCollector")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataCollector")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataCollector")
    process.gctDigis.inputLabel = cms.InputTag("rawDataCollector")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataCollector")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataCollector")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataCollector")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")


    process.load("Configuration.StandardSequences.Reconstruction_cff")
    # Offline Beam Spot
    process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

    process.dqmBeamMonitor.OnlineMode = True              
    process.dqmBeamMonitor.resetEveryNLumi = 5
    process.dqmBeamMonitor.resetPVEveryNLumi = 5
    process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 20
    process.dqmBeamMonitor.PVFitter.minVertexNdf = 10
  

    if (runFirstStepTrk): # for first Step Tracking
        print "[beam_dqm_sourceclient-live_cfg]:: firstStepTracking"
        # Import TrackerLocalReco sequence
        process.load('RecoLocalTracker.Configuration.RecoLocalTracker_cff')
        # Import MeasurementTrackerEvents used during patter recognition
        process.load('RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi')
        #Import stuff to run the initial step - PreSplitting - of the
        #iterative tracking and remove Calo-related sequences
        process.load('RecoTracker.IterativeTracking.InitialStepPreSplitting_cff')
        process.InitialStepPreSplitting.remove(process.initialStepTrackRefsForJetsPreSplitting)
        process.InitialStepPreSplitting.remove(process.caloTowerForTrkPreSplitting)
        process.InitialStepPreSplitting.remove(process.ak4CaloJetsForTrkPreSplitting)
        process.InitialStepPreSplitting.remove(process.jetsForCoreTrackingPreSplitting)
        process.InitialStepPreSplitting.remove(process.siPixelClusters)
        process.InitialStepPreSplitting.remove(process.siPixelRecHits)
        process.InitialStepPreSplitting.remove(process.MeasurementTrackerEvent)
        process.InitialStepPreSplitting.remove(process.siPixelClusterShapeCache)

        #Increase pT threashold at seeding stage (not so accurate)                                                                                      
        process.initialStepSeedsPreSplitting.RegionFactoryPSet.RegionPSet.ptMin = 0.9

        # some inputs to BeamMonitor
        process.dqmBeamMonitor.BeamFitter.TrackCollection = 'initialStepTracksPreSplitting'         
        process.dqmBeamMonitor.primaryVertex = 'firstStepPrimaryVerticesPreSplitting'
        process.dqmBeamMonitor.PVFitter.VertexCollection = 'firstStepPrimaryVerticesPreSplitting'

        process.dqmBeamMonitor.PVFitter.errorScale = 0.95 #keep checking this with new release expected close to 1

        process.tracking_FirstStep  = cms.Sequence( process.siPixelDigis*
                                                     process.siStripDigis*
                                                     process.pixeltrackerlocalreco*
                                                     process.striptrackerlocalreco*
                                                     process.offlineBeamSpot*                          
                                                     process.MeasurementTrackerEventPreSplitting*
                                                     process.siPixelClusterShapeCachePreSplitting*
                                                     process.InitialStepPreSplitting
                                                     )
    else: # pixel tracking
        print "[beam_dqm_sourceclient-live_cfg]:: pixelTracking"
        process.load("RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi")
        #pixel  track/vertices reco
        process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
        process.pixelVertices.TkFilterParameters.minPt = process.pixelTracks.RegionFactoryPSet.RegionPSet.ptMin
        process.offlinePrimaryVertices.TrackLabel = cms.InputTag("pixelTracks")

        process.dqmBeamMonitor.PVFitter.errorScale = 1.25 #keep checking this with new release expected close to 1.2
     

        from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
        process.PixelLayerTriplets.BPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
        process.PixelLayerTriplets.FPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
        from RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff import *
        process.pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.clusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCachePreSplitting')
 
        process.tracking_FirstStep  = cms.Sequence(process.siPixelDigis* 
                                                   process.offlineBeamSpot*
                                                   process.siPixelClustersPreSplitting*
                                                   process.siPixelRecHitsPreSplitting*
                                                   process.siPixelClusterShapeCachePreSplitting*
                                                   process.recopixelvertexing
                                                  )
 

    #TriggerName for selecting pv for DIP publication, NO wildcard needed here
    #it will pick all triggers which has these strings in theri name
    process.dqmBeamMonitor.jetTrigger  = cms.untracked.vstring("HLT_PAZeroBias_v",
                                                               "HLT_ZeroBias_v", 
                                                               "HLT_QuadJet")

    process.dqmBeamMonitor.hltResults = cms.InputTag("TriggerResults","","HLT")


    process.p = cms.Path(process.scalersRawToDigi
                         *process.dqmTKStatus
                         *process.hltTriggerTypeFilter
                         *process.dqmcommon
                         *process.tracking_FirstStep
                         *process.monitor
                         *process.BeamSpotProblemModule)





#--------------------------------------------------
# Heavy Ion Stuff
#--------------------------------------------------
if (process.runType.getRunType() == process.runType.hi_run):

    print "beam_dqm_sourceclient-live_cfg:Running HI"
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")

    #----------------------------
    # Event Source
    #-----------------------------

    process.dqmBeamMonitor.OnlineMode = True                  ## in MC the LS are not ordered??
    process.dqmBeamMonitor.resetEveryNLumi = 10
    process.dqmBeamMonitor.resetPVEveryNLumi = 10
    process.dqmBeamMonitor.BeamFitter.MinimumTotalLayers = 3   ## using pixel triplets
    process.dqmBeamMonitor.PVFitter.minVertexNdf = 10
    process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 20
    process.dqmBeamMonitor.PVFitter.errorScale = 1.25

    process.dqmBeamMonitor.jetTrigger  = cms.untracked.vstring("HLT_HI")

    process.dqmBeamMonitor.hltResults = cms.InputTag("TriggerResults","","HLT")


    ## Load Heavy Ion Sequence
    process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff") ## HI sequences

    # Select events based on the pixel cluster multiplicity
    import  HLTrigger.special.hltPixelActivityFilter_cfi
    process.multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
      inputTag  = cms.InputTag('siPixelClusters'),
      minClusters = cms.uint32(150),
      maxClusters = cms.uint32(50000)
      )

    process.filter_step = cms.Sequence( process.siPixelDigis
                                       *process.siPixelClusters
                                       #*process.multFilter
                                  )

    process.HIRecoForDQM = cms.Sequence( process.siPixelDigis
                                    *process.siPixelClusters
                                    *process.siPixelRecHits
                                    *process.offlineBeamSpot
                                    *process.hiPixelVertices
                                    *process.hiPixel3PrimTracks
                                   )

    # use HI pixel tracking and vertexing
    process.dqmBeamMonitor.BeamFitter.TrackCollection = cms.untracked.InputTag('hiPixel3PrimTracks')
    process.dqmBeamMonitorBx.BeamFitter.TrackCollection = cms.untracked.InputTag('hiPixel3PrimTracks')
    process.dqmBeamMonitor.primaryVertex = cms.untracked.InputTag('hiSelectedVertex')
    process.dqmBeamMonitor.PVFitter.VertexCollection = cms.untracked.InputTag('hiSelectedVertex')


    # make pixel vertexing less sensitive to incorrect beamspot
    process.hiPixel3ProtoTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.2
    process.hiPixel3ProtoTracks.RegionFactoryPSet.RegionPSet.fixedError = 0.5
    process.hiSelectedProtoTracks.maxD0Significance = 100
    process.hiPixelAdaptiveVertex.TkFilterParameters.maxD0Significance = 100
    process.hiPixelAdaptiveVertex.vertexCollections.useBeamConstraint = False
    #not working due to wrong tag of reco
    process.hiPixelAdaptiveVertex.vertexCollections.maxDistanceToBeam = 1.0



    
    process.p = cms.Path(process.scalersRawToDigi
                        *process.dqmTKStatus
                        *process.hltTriggerTypeFilter
                        *process.filter_step
                        *process.HIRecoForDQM
                        *process.dqmcommon
                        *process.monitor
                        *process.BeamSpotProblemModule)

