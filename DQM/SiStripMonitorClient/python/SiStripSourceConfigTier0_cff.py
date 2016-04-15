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
SiStripMonitorDigi.TkHistoMapMedianChargeApvShots_On= False
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
SiStripMonitorClusterBPTX = SiStripMonitorCluster.clone()
SiStripMonitorClusterBPTX.Mod_On = False
SiStripMonitorClusterBPTX.TH1TotalNumberOfClusters.subdetswitchon   = True
SiStripMonitorClusterBPTX.TProfClustersApvCycle.subdetswitchon      = True
SiStripMonitorClusterBPTX.TProfTotalNumberOfClusters.subdetswitchon = True 
SiStripMonitorClusterBPTX.TrendVsLS = True
SiStripMonitorClusterBPTX.TH2CStripVsCpixel.globalswitchon       = True
SiStripMonitorClusterBPTX.TH1MultiplicityRegions.globalswitchon  = True
SiStripMonitorClusterBPTX.TH1MainDiagonalPosition.globalswitchon = True
SiStripMonitorClusterBPTX.TH1StripNoise2ApvCycle.globalswitchon  = True
SiStripMonitorClusterBPTX.TH1StripNoise3ApvCycle.globalswitchon  = True
SiStripMonitorClusterBPTX.ClusterHisto = True
SiStripMonitorClusterBPTX.BPTXfilter = cms.PSet(
    andOr         = cms.bool( False ),
    dbLabel       = cms.string("SiStripDQMTrigger"),
    l1Algorithms = cms.vstring( 'L1Tech_BPTX_plus_AND_minus.v0', 'L1_ZeroBias' ),
    andOrL1       = cms.bool( True ),
    errorReplyL1  = cms.bool( True ),
    l1BeforeMask  = cms.bool( True ) # specifies, if the L1 algorithm decision should be read as before (true) or after (false) masking is applied. 
)
SiStripMonitorClusterBPTX.PixelDCSfilter = cms.PSet(
    andOr         = cms.bool( False ),
    dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
    dcsPartitions = cms.vint32 ( 28, 29),
    andOrDcs      = cms.bool( False ),
    errorReplyDcs = cms.bool( True ),
)
SiStripMonitorClusterBPTX.StripDCSfilter = cms.PSet(
    andOr         = cms.bool( False ),
    dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
    dcsPartitions = cms.vint32 ( 24, 25, 26, 27 ),
    andOrDcs      = cms.bool( False ),
    errorReplyDcs = cms.bool( True ),
)



# Clone for SiStripMonitorTrack for all PDs but Minimum Bias and Jet ####
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi 
SiStripMonitorTrackCommon = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackCommon.TrackProducer = 'generalTracks'
SiStripMonitorTrackCommon.Mod_On        = False

from DQM.SiStripMonitorClient.pset4GenericTriggerEventFlag_cfi import *
# Clone for SiStripMonitorTrack for Minimum Bias ####
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrackMB = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackMB.TrackProducer = 'generalTracks'
SiStripMonitorTrackMB.Mod_On        = False
SiStripMonitorTrackMB.andOr         = genericTriggerEventFlag4fullTrackerAndHLTdb.andOr
SiStripMonitorTrackMB.dbLabel       = genericTriggerEventFlag4fullTrackerAndHLTdb.dbLabel
SiStripMonitorTrackMB.hltInputTag   = genericTriggerEventFlag4fullTrackerAndHLTdb.hltInputTag
SiStripMonitorTrackMB.hltPaths      = genericTriggerEventFlag4fullTrackerAndHLTdb.hltPaths 
SiStripMonitorTrackMB.hltDBKey      = genericTriggerEventFlag4fullTrackerAndHLTdb.hltDBKey
SiStripMonitorTrackMB.errorReplyHlt = genericTriggerEventFlag4fullTrackerAndHLTdb.errorReplyHlt
SiStripMonitorTrackMB.andOrHlt      = genericTriggerEventFlag4fullTrackerAndHLTdb.andOrHlt

### TrackerMonitorTrack defined and used only for MinimumBias ####
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResiduals.trajectoryInput = 'generalTracks'
MonitorTrackResiduals.Tracks          = 'generalTracks'
MonitorTrackResiduals.Mod_On          = False
MonitorTrackResiduals.andOr           = genericTriggerEventFlag4fullTrackerAndHLTdb.andOr
MonitorTrackResiduals.dbLabel         = genericTriggerEventFlag4fullTrackerAndHLTdb.dbLabel
MonitorTrackResiduals.hltInputTag     = genericTriggerEventFlag4fullTrackerAndHLTdb.hltInputTag
MonitorTrackResiduals.hltPaths        = genericTriggerEventFlag4fullTrackerAndHLTdb.hltPaths
MonitorTrackResiduals.hltDBKey        = genericTriggerEventFlag4fullTrackerAndHLTdb.hltDBKey
MonitorTrackResiduals.errorReplyHlt   = genericTriggerEventFlag4fullTrackerAndHLTdb.errorReplyHlt
MonitorTrackResiduals.andOrHlt        = genericTriggerEventFlag4fullTrackerAndHLTdb.andOrHlt

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
from DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi import *

# temporary patch in order to have BXlumi 
from RecoLuminosity.LumiProducer.lumiProducer_cff import *

# Sequence
#removed modules using TkDetMap service
#SiStripDQMTier0 = cms.Sequence(
#    APVPhases*consecutiveHEs*siStripFEDCheck
#    *MonitorTrackResiduals
#    *dqmInfoSiStrip)

#SiStripDQMTier0Common = cms.Sequence(
#    APVPhases*consecutiveHEs*siStripFEDCheck
#    *dqmInfoSiStrip)

#SiStripDQMTier0MinBias = cms.Sequence(
#    APVPhases*consecutiveHEs*siStripFEDCheck
#    *SiStripMonitorTrackMB*MonitorTrackResiduals
#    *dqmInfoSiStrip)

SiStripDQMTier0 = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorClusterBPTX
    *SiStripMonitorTrackCommon*MonitorTrackResiduals
    *dqmInfoSiStrip)

SiStripDQMTier0Common = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorClusterBPTX        
    *SiStripMonitorTrackCommon
    *dqmInfoSiStrip)

SiStripDQMTier0MinBias = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorClusterBPTX
    *SiStripMonitorTrackMB*MonitorTrackResiduals
    *dqmInfoSiStrip)



