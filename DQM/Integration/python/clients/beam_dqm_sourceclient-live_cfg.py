from __future__ import print_function
import FWCore.ParameterSet.Config as cms

# Define here the BeamSpotOnline record name,
# it will be used both in BeamMonitor setup and in payload creation/upload
BSOnlineRecordName = 'BeamSpotOnlineLegacyObjectsRcd'
BSOnlineTag = 'BeamSpotOnlineLegacy'
BSOnlineJobName = 'BeamSpotOnlineLegacy'
BSOnlineOmsServiceUrl = 'http://cmsoms-eventing.cms:9949/urn:xdaq-application:lid=100/getRunAndLumiSection'
useLockRecords = True

import sys
if 'runkey=hi_run' in sys.argv:
  from Configuration.Eras.Era_Run3_pp_on_PbPb_approxSiStripClusters_cff import Run3_pp_on_PbPb_approxSiStripClusters
  process = cms.Process("BeamMonitorLegacy", Run3_pp_on_PbPb_approxSiStripClusters)
else:
  from Configuration.Eras.Era_Run3_cff import Run3
  process = cms.Process("BeamMonitorLegacy", Run3)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    destinations = cms.untracked.vstring('cerr')
)

# switch
live = True # FIXME
unitTest = False

if 'unitTest=True' in sys.argv:
    live=False
    unitTest=True
    useLockRecords = False

#---------------
# Input sources
if unitTest:
    process.load("DQM.Integration.config.unittestinputsource_cfi")
    from DQM.Integration.config.unittestinputsource_cfi import options
elif live:
    process.load("DQM.Integration.config.inputsource_cfi")
    from DQM.Integration.config.inputsource_cfi import options
else:
    process.load("DQM.Integration.config.fileinputsource_cfi")
    from DQM.Integration.config.fileinputsource_cfi import options

#--------------------------
# HLT Filter
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1) # physics
)

#----------------------------
# DQM Live Environment
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamMonitorLegacy'
process.dqmSaver.tag           = 'BeamMonitorLegacy'
process.dqmSaver.runNumber     = options.runNumber
process.dqmSaverPB.tag         = 'BeamMonitorLegacy'
process.dqmSaverPB.runNumber   = options.runNumber

process.dqmEnvPixelLess = process.dqmEnv.clone(
  subSystemFolder = 'BeamMonitor_PixelLess'
)

# Configure tag and jobName if running Playback system
if process.isDqmPlayback.value :
    BSOnlineTag = BSOnlineTag + 'Playback'
    BSOnlineJobName = BSOnlineJobName + 'Playback'
    BSOnlineOmsServiceUrl = ''
    useLockRecords = False
#

#---------------
# Conditions
if (live):
    process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
    process.GlobalTag.DBParameters.authenticationPath = '.'
else:
    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
    from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
    process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')
    process.GlobalTag.DBParameters.authenticationPath = '.'
    # you may need to set manually the GT in the line below
    #process.GlobalTag.globaltag = '100X_upgrade2018_realistic_v10'


#--------------------------------------------------------
# Swap offline <-> online BeamSpot as in Express and HLT
import RecoVertex.BeamSpotProducer.onlineBeamSpotESProducer_cfi as _mod
process.BeamSpotESProducer = _mod.onlineBeamSpotESProducer.clone()
import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

#----------------------------
# BeamMonitor
process.load("DQM.BeamMonitor.BeamMonitor_Pixel_cff")
process.load("DQM.BeamMonitor.BeamSpotProblemMonitor_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")

if process.dqmRunConfig.type.value() == "production":
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
process.pixelTracksMonitor = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone(
   FolderName                = 'BeamMonitorLegacy/Tracking/pixelTracks',
   TrackProducer             = 'pixelTracks',
   allTrackProducer          = 'pixelTracks',
   beamSpot                  = "offlineBeamSpot",
   primaryVertex             = "pixelVertices",

   doAllPlots                = False,
   doLumiAnalysis            = False,
   doProfilesVsLS            = True,
   doDCAPlots                = True,
   doPlotsVsGoodPVtx         = True,

   doEffFromHitPatternVsPU   = False,
   doEffFromHitPatternVsBX   = True,
   doEffFromHitPatternVsLUMI = False,
   doPlotsVsLUMI             = True,
   doPlotsVsBX               = True,

   AbsDxyMax                 = 1.2,
   AbsDxyBin                 = 12,
   DxyMin                    = -1.2,
   DxyMax                    = 1.2,
   DxyBin                    = 60,

   Chi2NDFMax                = 35.,
   Chi2NDFMin                = 0.,
   Chi2NDFBin                = 70,

   VZBin                     = 124,
   VZMin                     = -62.,
   VZMax                     =  62.,

   TrackPtMin                =  0.,
   TrackPtMax                =  50.,
   TrackPtBin                =  250
)
#
process.tracks2monitor = cms.EDFilter('TrackSelector',
    src = cms.InputTag('pixelTracks'),
    cut = cms.string("")
)
process.tracks2monitor.src = 'pixelTracksHP'
process.tracks2monitor.cut = 'pt > 1 & abs(eta) < 2.4' 


#
process.selectedPixelTracksMonitor = process.pixelTracksMonitor.clone(
   FolderName       = 'BeamMonitorLegacy/Tracking/selectedPixelTracks',
   TrackProducer    = 'tracks2monitor',
   allTrackProducer = 'tracks2monitor'
)

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
                               * process.dqmSaver*process.dqmSaverPB)

#
process.monitor = cms.Sequence(process.dqmBeamMonitor
                             + process.selectedPixelTracksMonitorSequence)

#------------------------
# BeamSpotProblemMonitor

#
process.dqmBeamSpotProblemMonitor.monitorName = "BeamMonitorLegacy/BeamSpotProblemMonitor"
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

# Digitisation: produce the TCDS digis containing BST record
from EventFilter.OnlineMetaDataRawToDigi.tcdsRawToDigi_cfi import *
process.tcdsDigis = tcdsRawToDigi.clone()

#------------------------
# Set rawDataRepacker (HI and live) or rawDataCollector (for all the rest)
if (process.runType.getRunType() == process.runType.hi_run and live):
    rawDataInputTag = "rawDataRepacker"
else:
    rawDataInputTag = "rawDataCollector"

process.castorDigis.InputLabel           = rawDataInputTag
process.csctfDigis.producer              = rawDataInputTag 
process.dttfDigis.DTTF_FED_Source        = rawDataInputTag
process.ecalDigis.cpu.InputLabel         = rawDataInputTag
process.ecalPreshowerDigis.sourceTag     = rawDataInputTag
process.gctDigis.inputLabel              = rawDataInputTag
process.gtDigis.DaqGtInputTag            = rawDataInputTag
process.hcalDigis.InputLabel             = rawDataInputTag
process.muonCSCDigis.InputObjects        = rawDataInputTag
process.muonDTDigis.inputLabel           = rawDataInputTag
process.muonRPCDigis.InputLabel          = rawDataInputTag
process.scalersRawToDigi.scalersInputTag = rawDataInputTag
process.siPixelDigis.cpu.InputLabel      = rawDataInputTag
process.siStripDigis.ProductLabel        = rawDataInputTag
process.tcdsDigis.InputLabel             = rawDataInputTag

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.dqmBeamMonitor.OnlineMode = True
process.dqmBeamMonitor.monitorName = "BeamMonitorLegacy"
process.dqmBeamMonitor.recordName = BSOnlineRecordName
process.dqmBeamMonitor.useLockRecords = useLockRecords

process.dqmBeamMonitor.resetEveryNLumi   = 5 # was 10 for HI
process.dqmBeamMonitor.resetPVEveryNLumi = 5 # was 10 for HI

process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 20
process.dqmBeamMonitor.PVFitter.minVertexNdf = 10
process.dqmBeamMonitor.PVFitter.errorScale = 1.0

#----------------------------
# Pixel tracks/vertices reco
process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
from RecoVertex.PrimaryVertexProducer.OfflinePixel3DPrimaryVertices_cfi import *
process.pixelVertices = pixelVertices.clone(
  TkFilterParameters = dict( minPt = process.pixelTracksTrackingRegions.RegionPSet.ptMin)
)
#process.pixelTracksTrackingRegions.RegionPSet.ptMin = 0.1       # used in PilotBeam 2021, but not ok for standard collisions
process.pixelTracksTrackingRegions.RegionPSet.originRadius = 0.4 # used in PilotBeam 2021, to be checked again for standard collisions
# The following parameters were used in 2018 HI:
#process.pixelTracksTrackingRegions.RegionPSet.originHalfLength = 12
#process.pixelTracksTrackingRegions.RegionPSet.originXPos =  0.08
#process.pixelTracksTrackingRegions.RegionPSet.originYPos = -0.03
#process.pixelTracksTrackingRegions.RegionPSet.originZPos = 0.

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
# it will pick all triggers which have these strings in their name
process.dqmBeamMonitor.jetTrigger  = [
         "HLT_PAZeroBias_v", "HLT_ZeroBias_v", "HLT_QuadJet",
         "HLT_ZeroBias_",
         "HLT_HI",
         "HLT_PixelClusters"]

# for HI only: select events based on the pixel cluster multiplicity
if (process.runType.getRunType() == process.runType.hi_run):
    import HLTrigger.special.hltPixelActivityFilter_cfi
    process.multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
        inputTag  = 'siPixelClustersPreSplitting',
        minClusters = 150,
        maxClusters = 50000 # was 10000
    )
       
    process.filter_step = cms.Sequence( process.siPixelDigis
                                      * process.siPixelClustersPreSplitting
                                      * process.multFilter
    )

process.dqmBeamMonitor.hltResults = "TriggerResults::HLT"

#---------
# Upload BeamSpotOnlineObject (LegacyRcd) to CondDB
if unitTest == False:
    process.OnlineDBOutputService = cms.Service("OnlineDBOutputService",

        DBParameters = cms.PSet(
                                messageLevel = cms.untracked.int32(0),
                                authenticationPath = cms.untracked.string('.')
                            ),

        # Upload to CondDB
        connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS'),
        preLoadConnectionString = cms.untracked.string('frontier://FrontierProd/CMS_CONDITIONS'),

        runNumber = cms.untracked.uint64(options.runNumber),
        omsServiceUrl = cms.untracked.string(BSOnlineOmsServiceUrl),
        latency = cms.untracked.uint32(2),
        autoCommit = cms.untracked.bool(True),
        saveLogsOnDB = cms.untracked.bool(True),
        jobName = cms.untracked.string(BSOnlineJobName), # name of the DB log record
        toPut = cms.VPSet(cms.PSet(
            record = cms.string(BSOnlineRecordName),
            tag = cms.string(BSOnlineTag),
            timetype = cms.untracked.string('Lumi'),
            onlyAppendUpdatePolicy = cms.untracked.bool(True)
        )),
        frontierKey = cms.untracked.string(options.runUniqueKey)
    )

else:
    process.OnlineDBOutputService = cms.Service("OnlineDBOutputService",

        DBParameters = cms.PSet(
                                messageLevel = cms.untracked.int32(0),
                                authenticationPath = cms.untracked.string('.')
                            ),

        # Upload to CondDB
        connect = cms.string('sqlite_file:BeamSpotOnlineLegacy.db'),
        preLoadConnectionString = cms.untracked.string('sqlite_file:BeamSpotOnlineLegacy.db'),
        runNumber = cms.untracked.uint64(options.runNumber),
        lastLumiFile = cms.untracked.string('last_lumi.txt'),
        latency = cms.untracked.uint32(2),
        autoCommit = cms.untracked.bool(True),
        toPut = cms.VPSet(cms.PSet(
            record = cms.string(BSOnlineRecordName),
            tag = cms.string(BSOnlineTag),
            timetype = cms.untracked.string('Lumi'),
            onlyAppendUpdatePolicy = cms.untracked.bool(True)
        )),
        frontierKey = cms.untracked.string(options.runUniqueKey)
    )
print("Configured frontierKey", options.runUniqueKey)

#--------
# Do no run on events with pixel or strip with HV off

process.stripTrackerHVOn = cms.EDFilter( "DetectorStateFilter",
    DCSRecordLabel = cms.untracked.InputTag( "onlineMetaDataDigis" ),
    DcsStatusLabel = cms.untracked.InputTag( "scalersRawToDigi" ),
    DebugOn = cms.untracked.bool( False ),
    DetectorType = cms.untracked.string( "sistrip" )
)

process.pixelTrackerHVOn = cms.EDFilter( "DetectorStateFilter",
    DCSRecordLabel = cms.untracked.InputTag( "onlineMetaDataDigis" ),
    DcsStatusLabel = cms.untracked.InputTag( "scalersRawToDigi" ),
    DebugOn = cms.untracked.bool( False ),
    DetectorType = cms.untracked.string( "pixel" )
)

#---------
# Final path
if (not process.runType.getRunType() == process.runType.hi_run):
    process.p = cms.Path(process.scalersRawToDigi
                       * process.tcdsDigis
                       * process.onlineMetaDataDigis
                       * process.pixelTrackerHVOn
                       * process.stripTrackerHVOn
                       * process.dqmTKStatus
                       * process.hltTriggerTypeFilter
                       * process.dqmcommon
                       * process.tracking_FirstStep
                       * process.monitor
                       * process.BeamSpotProblemModule)
else:
    process.p = cms.Path(process.scalersRawToDigi
                       * process.tcdsDigis
                       * process.onlineMetaDataDigis
                       * process.pixelTrackerHVOn
                       * process.stripTrackerHVOn
                       * process.dqmTKStatus
                       * process.hltTriggerTypeFilter
                       * process.filter_step # the only extra: pix-multi filter
                       * process.dqmcommon
                       * process.tracking_FirstStep
                       * process.monitor
                       * process.BeamSpotProblemModule)

print("Global Tag used:", process.GlobalTag.globaltag.value())
print("Final Source settings:", process.source)

