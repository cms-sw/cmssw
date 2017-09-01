import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("BeamMonitor", eras.Run2_2017)

#----------------------------------------------                                                                                                                                    
# Switch to change between firstStep and Pixel
#-----------------------------------------------

runFirstStepTrk = False

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
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
process.siPixelClusterShapeCachePreSplitting = siPixelClusterShapeCache.clone(
  src = 'siPixelClustersPreSplitting'
)
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi")


#---------------
# Calibration
#---------------
# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')

# Change Beam Monitor variables
if process.dqmRunConfig.type.value() is "production":
  process.dqmBeamMonitor.BeamFitter.WriteAscii = True
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults.txt'
else:
  process.dqmBeamMonitor.BeamFitter.WriteAscii = False
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmdev/BeamMonitorDQM/BeamFitResults.txt'


#----------------------------
# TrackingMonitor
#-----------------------------
# first, tracks selection as in https://github.com/cms-sw/cmssw/blob/master/DQM/BeamMonitor/python/BeamMonitor_Pixel_cff.py
# -MinimumPt = cms.untracked.double(1.0),
# -MaximumEta = cms.untracked.double(2.4),
# -MaximumImpactParameter = cms.untracked.double(1.0),
# -MaximumZ = cms.untracked.double(60),
# -MinimumTotalLayers = cms.untracked.int32(3),
# -MinimumPixelLayers = cms.untracked.int32(3),
# -MaximumNormChi2 = cms.untracked.double(30.0),

#### while hitPattern().trackerLayersWithMeasurement() and hitPattern().pixelLayersWithMeasurement() !!!
#### =>
#### select tracks based on MaximumImpactParameter, MaximumZ, MinimumTotalLayers, MinimumPixelLayers and MaximumNormChi2
process.pixelTracksCutClassifier = cms.EDProducer( "TrackCutClassifier",
    src = cms.InputTag( "pixelTracks" ),
    GBRForestLabel = cms.string( "" ),
    beamspot = cms.InputTag( "offlineBeamSpot" ),
#    vertices = cms.InputTag( "pixelVertices" ),
    vertices = cms.InputTag( "" ),
    qualityCuts = cms.vdouble( -0.7, 0.1, 0.7 ),
    mva = cms.PSet(
      minPixelHits = cms.vint32( 0, 3, 3 ),
      maxDzWrtBS = cms.vdouble( 3.40282346639E38, 3.40282346639E38, 60.0 ),
      dr_par = cms.PSet(
        d0err = cms.vdouble( 0.003, 0.003, 3.40282346639E38 ),
        dr_par2 = cms.vdouble( 0.3, 0.3, 3.40282346639E38 ),
        dr_par1 = cms.vdouble( 0.4, 0.4, 3.40282346639E38 ),
        dr_exp = cms.vint32( 4, 4, 4 ),
        d0err_par = cms.vdouble( 0.001, 0.001, 3.40282346639E38 )
      ),
      maxLostLayers = cms.vint32( 99, 99, 99 ),
      min3DLayers = cms.vint32( 0, 2, 3 ),
      dz_par = cms.PSet(
        dz_par1 = cms.vdouble( 0.4, 0.4, 3.40282346639E38 ),
        dz_par2 = cms.vdouble( 0.35, 0.35, 3.40282346639E38 ),
        dz_exp = cms.vint32( 4, 4, 4 )
      ),
      minNVtxTrk = cms.int32( 3 ),
      maxDz = cms.vdouble( 3.40282346639E38, 3.40282346639E38, 3.40282346639E38 ),
      minNdof = cms.vdouble( 1.0E-5, 1.0E-5, 1.0E-5 ),
      maxChi2 = cms.vdouble( 9999., 9999., 30.0 ),
      maxDr = cms.vdouble( 99., 99., 1. ),
      minLayers = cms.vint32( 0, 2, 3 )
    ),
    ignoreVertices = cms.bool( True ),
    GBRForestFileName = cms.string( "" )
)
process.pixelTracksHP = cms.EDProducer( "TrackCollectionFilterCloner",
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False ),
    originalSource = cms.InputTag( "pixelTracks" ),
    originalQualVals = cms.InputTag( 'pixelTracksCutClassifier','QualityMasks' ),
    originalMVAVals = cms.InputTag( 'pixelTracksCutClassifier','MVAValues' )
)

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
process.pixelTracksMonitor = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
process.pixelTracksMonitor.FolderName                = 'BeamMonitor/Tracking/pixelTracks'
if (runFirstStepTrk):
  process.pixelTracksMonitor.TrackProducer             = 'initialStepTracksPreSplitting'
  process.pixelTracksMonitor.allTrackProducer          = 'initialStepTracksPreSplitting'
else:
  process.pixelTracksMonitor.TrackProducer             = 'pixelTracks'
  process.pixelTracksMonitor.allTrackProducer          = 'pixelTracks'
process.pixelTracksMonitor.beamSpot                  = "offlineBeamSpot"
process.pixelTracksMonitor.primaryVertex             = "pixelVertices"
process.pixelTracksMonitor.doAllPlots                = cms.bool(False)
process.pixelTracksMonitor.doLumiAnalysis            = cms.bool(False)
process.pixelTracksMonitor.doProfilesVsLS            = cms.bool(True)
process.pixelTracksMonitor.doDCAPlots                = cms.bool(True)
process.pixelTracksMonitor.doPlotsVsGoodPVtx         = cms.bool(True)
process.pixelTracksMonitor.doEffFromHitPatternVsPU   = cms.bool(False)
process.pixelTracksMonitor.doEffFromHitPatternVsBX   = cms.bool(True)
process.pixelTracksMonitor.doEffFromHitPatternVsLUMI = cms.bool(False)
process.pixelTracksMonitor.doPlotsVsGoodPVtx         = cms.bool(True)
process.pixelTracksMonitor.doPlotsVsLUMI             = cms.bool(True)
process.pixelTracksMonitor.doPlotsVsBX               = cms.bool(True)
process.pixelTracksMonitor.AbsDxyMax  =   1.2
process.pixelTracksMonitor.AbsDxyBin  =  12
process.pixelTracksMonitor.DxyMin     =  -1.2
process.pixelTracksMonitor.DxyMax     =   1.2
process.pixelTracksMonitor.DxyBin     =  60
process.pixelTracksMonitor.Chi2NDFMax =  35.
process.pixelTracksMonitor.Chi2NDFMin =   0.
process.pixelTracksMonitor.Chi2NDFBin =  70
process.pixelTracksMonitor.VZBin      = 124
process.pixelTracksMonitor.VZMin      = -62.
process.pixelTracksMonitor.VZMax      =  62.
process.pixelTracksMonitor.TrackPtMin =   0.
process.pixelTracksMonitor.TrackPtMax =  50.
process.pixelTracksMonitor.TrackPtBin = 250

process.tracks2monitor = cms.EDFilter('TrackSelector',
    src = cms.InputTag('pixelTracks'),
    cut = cms.string("")
)
process.tracks2monitor.src = 'pixelTracksHP'
process.tracks2monitor.cut = 'pt > 1 & abs(eta) < 2.4' # & normalizedChi2 < 30. abs(dxy) < 1. abs(dz) < 60.'


process.selectedPixelTracksMonitor = process.pixelTracksMonitor.clone()
process.selectedPixelTracksMonitor.FolderName                = 'BeamMonitor/Tracking/selectedPixelTracks'
if (runFirstStepTrk):
  process.selectedPixelTracksMonitor.TrackProducer             = 'initialStepTracksPreSplitting'
  process.selectedPixelTracksMonitor.allTrackProducer          = 'initialStepTracksPreSplitting'
else:
  process.selectedPixelTracksMonitor.TrackProducer             = 'tracks2monitor'
  process.selectedPixelTracksMonitor.allTrackProducer          = 'tracks2monitor'

process.selectedPixelTracksMonitorSequence = cms.Sequence(
  process.pixelTracksCutClassifier
  + process.pixelTracksHP
  + process.tracks2monitor
  + process.selectedPixelTracksMonitor
)


## TKStatus
process.dqmTKStatus = cms.EDAnalyzer("TKStatus",
        BeamFitter = cms.PSet(
        DIPFileName = process.dqmBeamMonitor.BeamFitter.DIPFileName
        )
)


process.dqmcommon = cms.Sequence(process.dqmEnv
                                *process.dqmSaver)

process.monitor = cms.Sequence(process.dqmBeamMonitor + process.selectedPixelTracksMonitorSequence)


#------------------------------------------------------------
# BeamSpotProblemMonitor Modules
#-----------------------------------------------------------
process.dqmBeamSpotProblemMonitor.monitorName       = "BeamMonitor/BeamSpotProblemMonitor"
process.dqmBeamSpotProblemMonitor.AlarmONThreshold  = 15 # 10
process.dqmBeamSpotProblemMonitor.AlarmOFFThreshold = 17 # 12
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
    process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataCollector")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataCollector")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")


    process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

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
        # 2016-11-28 MK FIXME: I suppose I should migrate the lines below following the "new seeding framework"
        #
        # if z is very far due to bad fit
        process.initialStepSeedsPreSplitting.RegionFactoryPSet.RegionPSet.originRadius = 1.5
        process.initialStepSeedsPreSplitting.RegionFactoryPSet.RegionPSet.originHalfLength = cms.double(30.0)
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


        #pixel  track/vertices reco
        process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
        process.pixelTracksTrackingRegions.RegionPSet.originRadius = 0.4
        process.pixelTracksTrackingRegions.RegionPSet.originHalfLength = 12
        process.pixelTracksTrackingRegions.RegionPSet.originXPos = 0.08
        process.pixelTracksTrackingRegions.RegionPSet.originYPos = -0.03
        process.pixelTracksTrackingRegions.RegionPSet.originZPos = 0.
        process.pixelVertices.TkFilterParameters.minPt = process.pixelTracksTrackingRegions.RegionPSet.ptMin

        process.dqmBeamMonitor.PVFitter.errorScale = 1.22 #keep checking this with new release expected close to 1.2
 
        process.tracking_FirstStep  = cms.Sequence(process.siPixelDigis* 
                                                   process.siStripDigis *
                                                   process.striptrackerlocalreco *
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
    process.dqmBeamMonitor.BeamFitter.MinimumPixelLayers = 3
    process.dqmBeamMonitor.BeamFitter.MaximumNormChi2    = 30.0

    process.dqmBeamMonitor.PVFitter.minVertexNdf = 4
    process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 20
    process.dqmBeamMonitor.PVFitter.errorScale = 1.25       ## taken from 2012 pixel vtx studies

    process.dqmBeamMonitor.jetTrigger  = cms.untracked.vstring("HLT_HI")
    process.dqmBeamMonitor.hltResults = cms.InputTag("TriggerResults","","HLT")
    process.dqmBeamSpotProblemMonitor.pixelTracks = 'hiPixel3PrimTracks'

    # copy from online
    import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
    offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

    ## Load Heavy Ion Sequence
    process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff") ## HI sequences
    process.load('RecoLocalTracker.Configuration.RecoLocalTrackerHeavyIons_cff')
    from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
    from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
    siPixelClusterShapeCachePreSplitting = siPixelClusterShapeCache.clone(
       src = 'siPixelClustersPreSplitting'
    )  
       
    # Select events based on the pixel cluster multiplicity
    import  HLTrigger.special.hltPixelActivityFilter_cfi
    process.multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
      inputTag  = cms.InputTag('siPixelClustersPreSplitting'),
      minClusters = cms.uint32(150),
      maxClusters = cms.uint32(10000) #80% efficiency in HI MC
      )
       
    process.filter_step = cms.Sequence( process.siPixelDigis
                                       *process.siPixelClustersPreSplitting
                                       *process.multFilter
                                  )
       
    from RecoHI.HiTracking.HIPixelVerticesPreSplitting_cff import *
       
    process.PixelLayerTriplets.BPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')
    process.PixelLayerTriplets.FPix.HitProducer = cms.string('siPixelRecHitsPreSplitting')

    process.hiPixel3PrimTracksFilter = process.hiFilter.clone(
        VertexCollection = "hiSelectedVertexPreSplitting",
        chi2 = 1000.0,
        clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting",
        lipMax = 0.3,
        nSigmaLipMaxTolerance = 0,
        nSigmaTipMaxTolerance = 6.0,
        ptMin = 0.9,
        tipMax = 0,
    )
    process.hiPixel3PrimTracks.Filter = "hiPixel3PrimTracksFilter"
      
    process.hiPixel3PrimTracks.RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer'),
        RegionPSet = cms.PSet(
            VertexCollection = cms.InputTag("hiSelectedVertexPreSplitting"),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            fixedError = cms.double(0.2),
            nSigmaZ = cms.double(3.0),
            originRadius = cms.double(0.1),
            precise = cms.bool(True),
            ptMin = cms.double(0.9),
            sigmaZVertex = cms.double(3.0),
            useFixedError = cms.bool(True),
            useFoundVertices = cms.bool(True)
        )
        )
    process.hiPixel3PrimTracks.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.clusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCachePreSplitting')
    process.hiPixel3PrimTracks.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.clusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCachePreSplitting')
    ## From HI group     
    process.HIRecoForDQM = cms.Sequence( process.siPixelDigis
                                        *process.siStripDigis
                                        *process.offlineBeamSpot  #<-clone of onlineBS
                                        *process.pixeltrackerlocalreco
                                        *process.striptrackerlocalreco             
                                        *process.MeasurementTrackerEventPreSplitting
                                        *process.siPixelClusterShapeCachePreSplitting
                                        *process.hiPixelVerticesPreSplitting
                                        *process.PixelLayerTriplets 
                                        *process.pixelFitterByHelixProjections
                                        *process.hiPixel3PrimTracksFilter
                                        *process.hiPixel3PrimTracks
                                       )
      
    # use HI pixel tracking and vertexing
    process.dqmBeamMonitor.BeamFitter.TrackCollection = cms.untracked.InputTag('hiPixel3PrimTracks')
    process.dqmBeamMonitor.primaryVertex = cms.untracked.InputTag('hiSelectedVertexPreSplitting')
    process.dqmBeamMonitor.PVFitter.VertexCollection = cms.untracked.InputTag('hiSelectedVertexPreSplitting')

      
    # make pixel VERTEXING less sensitive to incorrect beamspot
    process.hiPixel3ProtoTracksPreSplitting.RegionFactoryPSet.RegionPSet.originRadius = 0.2 #default 0.2
    process.hiPixel3ProtoTracksPreSplitting.RegionFactoryPSet.RegionPSet.fixedError = 0.5   #default 3.0
    process.hiSelectedProtoTracksPreSplitting.maxD0Significance = 100                       #default 5.0
    process.hiPixelAdaptiveVertexPreSplitting.TkFilterParameters.maxD0Significance = 100    #default 3.0
    process.hiPixelAdaptiveVertexPreSplitting.vertexCollections.useBeamConstraint = False  #default False
    process.hiPixelAdaptiveVertexPreSplitting.vertexCollections.maxDistanceToBeam = 1.0    #default 0.1
      
      
    process.p = cms.Path(process.scalersRawToDigi
                        *process.dqmTKStatus
                        *process.hltTriggerTypeFilter
                        *process.filter_step
                        *process.HIRecoForDQM
                        *process.dqmcommon
                        *process.monitor
                        *process.BeamSpotProblemModule)
                                                        

