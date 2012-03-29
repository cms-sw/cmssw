import FWCore.ParameterSet.Config as cms

# FED integrity Check
from DQM.SiStripMonitorHardware.siStripFEDCheck_cfi import *
siStripFEDCheck.HistogramUpdateFrequency = 0
siStripFEDCheck.DoPayloadChecks          = True
siStripFEDCheck.CheckChannelLengths      = True
siStripFEDCheck.CheckChannelPacketCodes  = True
siStripFEDCheck.CheckFELengths           = True
siStripFEDCheck.CheckChannelStatus       = True

# FED Monitoring
from DQM.SiStripMonitorHardware.siStripFEDMonitor_Tier0_cff import *

# SiStripMonitorDigi ####
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi import *
SiStripMonitorDigi.Mod_On = False
SiStripMonitorDigi.TProfDigiApvCycle.subdetswitchon = True

# APV shots monitoring
SiStripMonitorDigi.TkHistoMapNApvShots_On = True 
SiStripMonitorDigi.TkHistoMapNStripApvShots_On= True
SiStripMonitorDigi.TkHistoMapMedianChargeApvShots_On= True
SiStripMonitorDigi.TH1NApvShots.subdetswitchon = True
SiStripMonitorDigi.TH1NApvShots.globalswitchon = True
SiStripMonitorDigi.TH1ChargeMedianApvShots.subdetswitchon = True
SiStripMonitorDigi.TH1ChargeMedianApvShots.globalswitchon = True
SiStripMonitorDigi.TH1NStripsApvShots.subdetswitchon = True
SiStripMonitorDigi.TH1NStripsApvShots.globalswitchon = True
SiStripMonitorDigi.TH1ApvNumApvShots.subdetswitchon = True
SiStripMonitorDigi.TH1ApvNumApvShots.globalswitchon = True
SiStripMonitorDigi.TProfNShotsVsTime.subdetswitchon = True
SiStripMonitorDigi.TProfNShotsVsTime.globalswitchon = True
SiStripMonitorDigi.TProfGlobalNShots.globalswitchon = True

# SiStripMonitorCluster ####
from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *
SiStripMonitorCluster.Mod_On = False
SiStripMonitorCluster.TProfClustersApvCycle.subdetswitchon = True
SiStripMonitorCluster.TProfTotalNumberOfClusters.subdetswitchon = True
SiStripMonitorCluster.TH2CStripVsCpixel.globalswitchon=True
SiStripMonitorCluster.TH1MultiplicityRegions.globalswitchon=True
SiStripMonitorCluster.TH1MainDiagonalPosition.globalswitchon=True
SiStripMonitorCluster.TH1StripNoise2ApvCycle.globalswitchon=True
SiStripMonitorCluster.TH1StripNoise3ApvCycle.globalswitchon=True
SiStripMonitorCluster.ClusterHisto = True

# Clone for SiStripMonitorTrack for all PDs but Minimum Bias and Jet ####
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi 
SiStripMonitorTrackCommon = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackCommon.TrackProducer = 'generalTracks'
SiStripMonitorTrackCommon.Mod_On        = False

# Clone for SiStripMonitorTrack for Minimum Bias ####
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrackMB = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackMB.TrackProducer = 'generalTracks'
SiStripMonitorTrackMB.Mod_On        = False
SiStripMonitorTrackMB.andOr         = cms.bool( False )
SiStripMonitorTrackMB.dbLabel       = cms.string("SiStripDQMTrigger")
SiStripMonitorTrackMB.hltInputTag = cms.InputTag( "TriggerResults::HLT" )
SiStripMonitorTrackMB.hltPaths = cms.vstring("HLT_ZeroBias_*")
SiStripMonitorTrackMB.hltDBKey = cms.string("Tracker_MB")
SiStripMonitorTrackMB.errorReplyHlt  = cms.bool( False )
SiStripMonitorTrackMB.andOrHlt = cms.bool(True) # True:=OR; False:=AND

### TrackerMonitorTrack defined and used only for MinimumBias ####
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResiduals.trajectoryInput = 'generalTracks'
MonitorTrackResiduals.OutputMEsInRootFile = False
MonitorTrackResiduals.Mod_On        = False
MonitorTrackResiduals.andOr         = cms.bool( False )
MonitorTrackResiduals.dbLabel       = cms.string("SiStripDQMTrigger")
MonitorTrackResiduals.hltInputTag = cms.InputTag( "TriggerResults::HLT" )
MonitorTrackResiduals.hltPaths = cms.vstring("HLT_ZeroBias_*")
MonitorTrackResiduals.hltDBKey = cms.string("Tracker_MB")
MonitorTrackResiduals.errorReplyHlt  = cms.bool( False )
MonitorTrackResiduals.andOrHlt = cms.bool(True) 

# Clone for TrackingMonitor for all PDs but MinBias and Jet ####
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonCommon = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonCommon.FolderName    = 'Tracking/TrackParameters'
TrackerCollisionTrackMonCommon.andOr         = cms.bool( False )
TrackerCollisionTrackMonCommon.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionTrackMonCommon.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionTrackMonCommon.andOrDcs      = cms.bool( False )
TrackerCollisionTrackMonCommon.errorReplyDcs = cms.bool( True )

# Clone for TrackingMonitor for MinBias ###
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonMB = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonMB.FolderName    = 'Tracking/TrackParameters'
TrackerCollisionTrackMonMB.andOr         = cms.bool( False )
TrackerCollisionTrackMonMB.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionTrackMonMB.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionTrackMonMB.andOrDcs      = cms.bool( False )
TrackerCollisionTrackMonMB.errorReplyDcs = cms.bool( True )
TrackerCollisionTrackMonMB.dbLabel       = cms.string("SiStripDQMTrigger")
TrackerCollisionTrackMonMB.hltInputTag = cms.InputTag( "TriggerResults::HLT" )
TrackerCollisionTrackMonMB.hltPaths = cms.vstring("HLT_ZeroBias_*")
TrackerCollisionTrackMonMB.hltDBKey = cms.string("Tracker_MB")
TrackerCollisionTrackMonMB.errorReplyHlt  = cms.bool( False )
TrackerCollisionTrackMonMB.andOrHlt = cms.bool(True) 

from DQM.TrackingMonitor.TrackingMonitorSeedNumber_cff import *

# DQM Services
dqmInfoSiStrip = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('SiStrip')
)

# Services needed for TkHistoMap
TkDetMap = cms.Service("TkDetMap")
SiStripDetInfoFileReade = cms.Service("SiStripDetInfoFileReader")

# Event History Producer
from  DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi import *

# APV Phase Producer
from DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2011_cfi import *

# LogMessageMonitor ####
from DQM.TrackingMonitor.LogMessageMonitor_cff import *
# Clone for all PDs but MinBias ####
import DQM.TrackingMonitor.LogMessageMonitor_cff
TrackerCollisionIterTrackingLogMessageMonCommon = DQM.TrackingMonitor.LogMessageMonitor_cff.FullIterTrackingLogMessageMon.clone()
TrackerCollisionIterTrackingLogMessageMonCommon.andOr         = cms.bool( False )
TrackerCollisionIterTrackingLogMessageMonCommon.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionIterTrackingLogMessageMonCommon.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionIterTrackingLogMessageMonCommon.andOrDcs      = cms.bool( False )
TrackerCollisionIterTrackingLogMessageMonCommon.errorReplyDcs = cms.bool( True )

# Clone for MinBias ###
TrackerCollisionIterTrackingLogMessageMonMB = DQM.TrackingMonitor.LogMessageMonitor_cff.FullIterTrackingLogMessageMon.clone()
TrackerCollisionIterTrackingLogMessageMonMB.andOr         = cms.bool( False )
TrackerCollisionIterTrackingLogMessageMonMB.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionIterTrackingLogMessageMonMB.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionIterTrackingLogMessageMonMB.andOrDcs      = cms.bool( False )
TrackerCollisionIterTrackingLogMessageMonMB.errorReplyDcs = cms.bool( True )
TrackerCollisionIterTrackingLogMessageMonMB.dbLabel       = cms.string("SiStripDQMTrigger")
TrackerCollisionIterTrackingLogMessageMonMB.hltInputTag = cms.InputTag( "TriggerResults::HLT" )
TrackerCollisionIterTrackingLogMessageMonMB.hltPaths = cms.vstring("HLT_ZeroBias_*")
TrackerCollisionIterTrackingLogMessageMonMB.hltDBKey = cms.string("Tracker_MB")
TrackerCollisionIterTrackingLogMessageMonMB.errorReplyHlt  = cms.bool( False )
TrackerCollisionIterTrackingLogMessageMonMB.andOrHlt = cms.bool(True) 


# temporary patch in order to have BXlumi 
from RecoLuminosity.LumiProducer.lumiProducer_cff import *

from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector
# temporary test in order to temporary produce the "goodPrimaryVertexCollection"
goodOfflinePrimaryVertices = cms.EDFilter(
    "PrimaryVertexObjectFilter",
    filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) ),
    src=cms.InputTag('offlinePrimaryVertices')
)

# Sequence
SiStripDQMTier0 = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorCluster
    *SiStripMonitorTrackCommon*MonitorTrackResiduals
#    # temporary patch in order to have BXlumi
#    * lumiProducer
    # temporary test in order to have the "goodPrimaryVertexCollection"
    * goodOfflinePrimaryVertices *TrackerCollisionTrackMonCommon
#    * LocalRecoLogMessageMon * TrackerCollisionIterTrackingLogMessageMonCommon
    *TrackMonStep0*TrackMonStep1*TrackMonStep2*TrackMonStep3*TrackMonStep4*TrackMonStep5*TrackMonStep6
    *dqmInfoSiStrip)

SiStripDQMTier0Common = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorCluster
    *SiStripMonitorTrackCommon
#    # temporary patch in order to have BXlumi
#    * lumiProducer
#    # temporary test in order to have the "goodPrimaryVertexCollection"
    * goodOfflinePrimaryVertices *TrackerCollisionTrackMonCommon
#    * LocalRecoLogMessageMon * TrackerCollisionIterTrackingLogMessageMonCommon
    *TrackMonStep0*TrackMonStep1*TrackMonStep2*TrackMonStep3*TrackMonStep4*TrackMonStep5*TrackMonStep6
    *dqmInfoSiStrip)

SiStripDQMTier0MinBias = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorCluster
    *SiStripMonitorTrackMB*MonitorTrackResiduals
#    * lumiProducer
#    # temporary test in order to have the "goodPrimaryVertexCollection"
    * goodOfflinePrimaryVertices *TrackerCollisionTrackMonMB
#    * LocalRecoLogMessageMon * TrackerCollisionIterTrackingLogMessageMonMB
    *TrackMonStep0*TrackMonStep1*TrackMonStep2*TrackMonStep3*TrackMonStep4*TrackMonStep5*TrackMonStep6
    *dqmInfoSiStrip)

