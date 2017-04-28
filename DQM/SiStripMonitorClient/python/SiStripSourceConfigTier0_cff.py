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
SiStripMonitorDigi.TkHistoMapNStripApvShots_On= False
SiStripMonitorDigi.TkHistoMapMedianChargeApvShots_On= False
SiStripMonitorDigi.TH1NApvShots.subdetswitchon = True
SiStripMonitorDigi.TH1NApvShots.globalswitchon = True
SiStripMonitorDigi.TH1ChargeMedianApvShots.subdetswitchon = False
SiStripMonitorDigi.TH1ChargeMedianApvShots.globalswitchon = True
SiStripMonitorDigi.TH1NStripsApvShots.subdetswitchon = False
SiStripMonitorDigi.TH1NStripsApvShots.globalswitchon = False
SiStripMonitorDigi.TH1ApvNumApvShots.subdetswitchon = False
SiStripMonitorDigi.TH1ApvNumApvShots.globalswitchon = False
SiStripMonitorDigi.TProfNShotsVsTime.subdetswitchon = False
SiStripMonitorDigi.TProfNShotsVsTime.globalswitchon = False
SiStripMonitorDigi.TProfGlobalNShots.globalswitchon = True

from DQM.SiStripMonitorClient.pset4GenericTriggerEventFlag_cfi import *

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
SiStripMonitorClusterBPTX.BPTXfilter = genericTriggerEventFlag4L1bd
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

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(SiStripMonitorClusterBPTX, 
    BPTXfilter = dict(
        stage2 = cms.bool(True),
        l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
        l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
        ReadPrescalesFromFile = cms.bool(True)
    )
)

# refitter ### (FIXME rename, move)
from DQM.SiPixelMonitorTrack.RefitterForPixelDQM import *

# Clone for SiStripMonitorTrack for all PDs but Minimum Bias and Jet ####
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi 
SiStripMonitorTrackCommon = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackCommon.TrackProducer = 'generalTracks'
SiStripMonitorTrackCommon.Mod_On        = False
SiStripMonitorTrackCommon.TH1ClusterCharge.ringView = cms.bool( True )
SiStripMonitorTrackCommon.TH1ClusterStoNCorr.ringView = cms.bool( True )
SiStripMonitorTrackCommon.TH1ClusterPos.layerView = cms.bool( False )
SiStripMonitorTrackCommon.TH1ClusterPos.ringView = cms.bool( True )

# Clone for SiStripMonitorTrack for Minimum Bias ####
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrackMB = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackMB.TrackProducer = 'generalTracks'
SiStripMonitorTrackMB.Mod_On        = False
SiStripMonitorTrackMB.genericTriggerEventPSet = genericTriggerEventFlag4HLTdb
SiStripMonitorTrackMB.TH1ClusterCharge.ringView = cms.bool( True )
SiStripMonitorTrackMB.TH1ClusterStoNCorr.ringView = cms.bool( True )

# Clone for SiStripMonitorTrack for Isolated Bunches ####
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrackIB = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackIB.TrackProducer = 'generalTracks'
SiStripMonitorTrackIB.Mod_On        = False
SiStripMonitorTrackIB.genericTriggerEventPSet = genericTriggerEventFlag4HLTdbIB
SiStripMonitorTrackIB.TH1ClusterCharge.ringView = cms.bool( True )
SiStripMonitorTrackIB.TH1ClusterStoNCorr.ringView = cms.bool( True )
SiStripMonitorTrackIB.TkHistoMap_On = cms.bool(False)
SiStripMonitorTrackIB.TH1ClusterNoise.layerView = cms.bool(False) 
SiStripMonitorTrackIB.TH1ClusterWidth.layerView = cms.bool(False) 
SiStripMonitorTrackIB.TH1ClusterChargePerCM.ringView = cms.bool(False) 
SiStripMonitorTrackIB.TopFolderName = cms.string("SiStrip/IsolatedBunches")

### TrackerMonitorTrack defined and used only for MinimumBias ####
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResiduals.trajectoryInput = 'refittedForPixelDQM'
MonitorTrackResiduals.Tracks          = 'refittedForPixelDQM'
MonitorTrackResiduals.Mod_On        = False
MonitorTrackResiduals.genericTriggerEventPSet = genericTriggerEventFlag4HLTdb

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
    *SiStripMonitorTrackCommon*SiStripMonitorTrackIB*refittedForPixelDQM*MonitorTrackResiduals
    *dqmInfoSiStrip)
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

SiStripDQMTier0Common = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorClusterBPTX        
    *SiStripMonitorTrackCommon*SiStripMonitorTrackIB
    *dqmInfoSiStrip)

SiStripDQMTier0MinBias = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorClusterBPTX
    *SiStripMonitorTrackMB*SiStripMonitorTrackIB*refittedForPixelDQM*MonitorTrackResiduals
    *dqmInfoSiStrip)



