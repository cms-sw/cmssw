from __future__ import print_function
import FWCore.ParameterSet.Config as cms

#from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
#process = cms.Process("BeamMonitor", Run2_2018) FIXME
from Configuration.Eras.Era_Run2_2018_pp_on_AA_cff import Run2_2018_pp_on_AA
process = cms.Process("BeamMonitor", Run2_2018_pp_on_AA)

#
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    destinations = cms.untracked.vstring('cerr')
)

# switch
live = True # FIXME

#---------------
# Input sources
if (live):
    process.load("DQM.Integration.config.inputsource_cfi")
else:
    process.load("DQM.Integration.config.fileinputsource_cfi")

#--------------------------
# HLT Filter
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1) # physics
)

#----------------------------
# DQM Live Environment
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.tag           = 'BeamMonitor'

process.dqmEnvPixelLess = process.dqmEnv.clone()
process.dqmEnvPixelLess.subSystemFolder = 'BeamMonitor_PixelLess'

#---------------
# Conditions
if (live):
    process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
else:
    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
    from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
    process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')
    # you may need to set manually the GT in the line below
    process.GlobalTag.globaltag = '100X_upgrade2018_realistic_v10'

#----------------------------
# BeamMonitor
process.load("DQM.BeamMonitor.BeamMonitor_Pixel_cff")
process.load("DQM.BeamMonitor.BeamSpotProblemMonitor_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")

if process.dqmRunConfig.type.value() is "production":
  process.dqmBeamMonitor.BeamFitter.WriteAscii = True
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResultsOld.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResultsOld.txt'
else:
  process.dqmBeamMonitor.BeamFitter.WriteAscii = False
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResultsOld.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  if (live):
    process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmdev/BeamMonitorDQM/BeamFitResultsOld.txt'
  else:
    process.dqmBeamMonitor.BeamFitter.DIPFileName = 'BeamFitResultsOld.txt'

#----------------
# Setup tracking
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

#-----------------

# TrackingMonitor
process.pixelTracksCutClassifier = cms.EDProducer( "TrackCutClassifier",
    src = cms.InputTag( "pixelTracks" ),
    beamspot = cms.InputTag( "offlineBeamSpot" ),
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
)

#
process.pixelTracksHP = cms.EDProducer( "TrackCollectionFilterCloner",
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False ),
    originalSource = cms.InputTag( "pixelTracks" ),
    originalQualVals = cms.InputTag( 'pixelTracksCutClassifier','QualityMasks' ),
    originalMVAVals = cms.InputTag( 'pixelTracksCutClassifier','MVAValues' )
)

#-------------------------------------
# PixelTracksMonitor

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
process.pixelTracksMonitor = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()

process.pixelTracksMonitor.FolderName       = 'BeamMonitor/Tracking/pixelTracks'
process.pixelTracksMonitor.TrackProducer    = 'pixelTracks'
process.pixelTracksMonitor.allTrackProducer = 'pixelTracks'
process.pixelTracksMonitor.beamSpot         = "offlineBeamSpot"
process.pixelTracksMonitor.primaryVertex    = "pixelVertices"

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

process.pixelTracksMonitor.VZBin      =  124
process.pixelTracksMonitor.VZMin      =  -62.
process.pixelTracksMonitor.VZMax      =   62.

process.pixelTracksMonitor.TrackPtMin =    0.
process.pixelTracksMonitor.TrackPtMax =   50.
process.pixelTracksMonitor.TrackPtBin =  250

#
process.tracks2monitor = cms.EDFilter('TrackSelector',
    src = cms.InputTag('pixelTracks'),
    cut = cms.string("")
)
process.tracks2monitor.src = 'pixelTracksHP'
process.tracks2monitor.cut = 'pt > 1 & abs(eta) < 2.4' 


#
process.selectedPixelTracksMonitor = process.pixelTracksMonitor.clone()
process.selectedPixelTracksMonitor.FolderName       = 'BeamMonitor/Tracking/selectedPixelTracks'
process.selectedPixelTracksMonitor.TrackProducer    = 'tracks2monitor'
process.selectedPixelTracksMonitor.allTrackProducer = 'tracks2monitor'

process.selectedPixelTracksMonitorSequence = cms.Sequence(
    process.pixelTracksCutClassifier
  + process.pixelTracksHP
  + process.tracks2monitor
  + process.selectedPixelTracksMonitor
)


#---------------------------------
# Putting together combined paths

#
process.dqmTKStatus = cms.EDAnalyzer("TKStatus",
    BeamFitter = cms.PSet(
        DIPFileName = process.dqmBeamMonitor.BeamFitter.DIPFileName
    )
)

#
process.dqmcommon = cms.Sequence(process.dqmEnv
                               * process.dqmSaver)

#
process.monitor = cms.Sequence(process.dqmBeamMonitor
                             + process.selectedPixelTracksMonitorSequence)

#------------------------
# BeamSpotProblemMonitor

#
process.dqmBeamSpotProblemMonitor.monitorName = "BeamMonitor/BeamSpotProblemMonitor"
process.dqmBeamSpotProblemMonitor.AlarmONThreshold  = 15 # was 10
process.dqmBeamSpotProblemMonitor.AlarmOFFThreshold = 17 # was 12
process.dqmBeamSpotProblemMonitor.nCosmicTrk        = 10
process.dqmBeamSpotProblemMonitor.doTest            = False
process.dqmBeamSpotProblemMonitor.pixelTracks  = 'pixelTracks'

#
from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/BeamMonitor/test/BeamSpotAvailableTest.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    qtestOnEndLumi = cms.untracked.bool(True),
    testInEventloop = cms.untracked.bool(False),
    verboseQT =  cms.untracked.bool(True)                 
)

#
process.BeamSpotProblemModule = cms.Sequence(process.qTester
 	  	                           * process.dqmBeamSpotProblemMonitor)

# make it off for cosmic run
if ( process.runType.getRunType() == process.runType.cosmic_run or
     process.runType.getRunType() == process.runType.cosmic_run_stage1):
    process.dqmBeamSpotProblemMonitor.AlarmOFFThreshold = 5 # <AlarmONThreshold

#------------------------
# Process customizations
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)

#------------------------
# Set rawDataRepacker (HI and live) or rawDataCollector (for all the rest)
if (process.runType.getRunType() == process.runType.hi_run and live):
    rawDataInputTag = cms.InputTag("rawDataRepacker")
else:
    rawDataInputTag = cms.InputTag("rawDataCollector")

process.castorDigis.InputLabel           = rawDataInputTag
process.csctfDigis.producer              = rawDataInputTag 
process.dttfDigis.DTTF_FED_Source        = rawDataInputTag
process.ecalDigis.InputLabel             = rawDataInputTag
process.ecalPreshowerDigis.sourceTag     = rawDataInputTag
process.gctDigis.inputLabel              = rawDataInputTag
process.gtDigis.DaqGtInputTag            = rawDataInputTag
process.hcalDigis.InputLabel             = rawDataInputTag
process.muonCSCDigis.InputObjects        = rawDataInputTag
process.muonDTDigis.inputLabel           = rawDataInputTag
process.muonRPCDigis.InputLabel          = rawDataInputTag
process.scalersRawToDigi.scalersInputTag = rawDataInputTag
process.siPixelDigis.InputLabel          = rawDataInputTag
process.siStripDigis.ProductLabel        = rawDataInputTag

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.dqmBeamMonitor.OnlineMode = True

process.dqmBeamMonitor.resetEveryNLumi   = 5 # was 10 for HI
process.dqmBeamMonitor.resetPVEveryNLumi = 5 # was 10 for HI

process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 20
process.dqmBeamMonitor.PVFitter.minVertexNdf = 10
process.dqmBeamMonitor.PVFitter.errorScale = 1.22

#----------------------------
# Pixel tracks/vertices reco
process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
process.pixelTracksTrackingRegions.RegionPSet.originRadius = 0.4
process.pixelTracksTrackingRegions.RegionPSet.originHalfLength = 12
process.pixelTracksTrackingRegions.RegionPSet.originXPos =  0.08
process.pixelTracksTrackingRegions.RegionPSet.originYPos = -0.03
process.pixelTracksTrackingRegions.RegionPSet.originZPos = 0.

process.pixelVertices.TkFilterParameters.minPt = process.pixelTracksTrackingRegions.RegionPSet.ptMin

process.tracking_FirstStep = cms.Sequence(
      process.siPixelDigis 
    * process.siStripDigis
    * process.striptrackerlocalreco
    * process.offlineBeamSpot
    * process.siPixelClustersPreSplitting
    * process.siPixelRecHitsPreSplitting
    * process.siPixelClusterShapeCachePreSplitting
    * process.recopixelvertexing)

# triggerName for selecting pv for DIP publication, no wildcard needed here
# it will pick all triggers which has these strings in theri name
process.dqmBeamMonitor.jetTrigger  = cms.untracked.vstring(
         "HLT_PAZeroBias_v", "HLT_ZeroBias_v", "HLT_QuadJet",
         "HLT_ZeroBias_",
         "HLT_HI")

# for HI only: select events based on the pixel cluster multiplicity
if (process.runType.getRunType() == process.runType.hi_run):
    import HLTrigger.special.hltPixelActivityFilter_cfi
    process.multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
        inputTag  = cms.InputTag('siPixelClustersPreSplitting'),
        minClusters = cms.uint32(150),
        maxClusters = cms.uint32(50000) # was 10000
    )
       
    process.filter_step = cms.Sequence( process.siPixelDigis
                                      * process.siPixelClustersPreSplitting
                                      * process.multFilter
    )

process.dqmBeamMonitor.hltResults = cms.InputTag("TriggerResults","","HLT")

#---------
# Final path
if (not process.runType.getRunType() == process.runType.hi_run):
    process.p = cms.Path(process.scalersRawToDigi
                       * process.dqmTKStatus
                       * process.hltTriggerTypeFilter
                       * process.dqmcommon
                       * process.tracking_FirstStep
                       * process.monitor
                       * process.BeamSpotProblemModule)
else:
    process.p = cms.Path(process.scalersRawToDigi
                       * process.dqmTKStatus
                       * process.hltTriggerTypeFilter
                       * process.filter_step # the only extra: pix-multi filter
                       * process.dqmcommon
                       * process.tracking_FirstStep
                       * process.monitor
                       * process.BeamSpotProblemModule)

