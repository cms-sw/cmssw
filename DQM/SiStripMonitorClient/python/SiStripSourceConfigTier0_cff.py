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
SiStripMonitorClusterBPTX = SiStripMonitorCluster.clone(
    Mod_On = False,
    TH1TotalNumberOfClusters = SiStripMonitorCluster.TH1TotalNumberOfClusters.clone(
        subdetswitchon = True
    ),
    TProfClustersApvCycle = SiStripMonitorCluster.TProfClustersApvCycle.clone(
        subdetswitchon = True
    ),
    TProfTotalNumberOfClusters = SiStripMonitorCluster.TProfTotalNumberOfClusters.clone(
        subdetswitchon = True
    ),
    TrendVs10LS = False,
    TH2CStripVsCpixel = SiStripMonitorCluster.TH2CStripVsCpixel.clone(
        globalswitchon = True
    ),
    TH1MultiplicityRegions = SiStripMonitorCluster.TH1MultiplicityRegions.clone(
        globalswitchon = True
    ),
    TH1MainDiagonalPosition = SiStripMonitorCluster.TH1MainDiagonalPosition.clone(
        globalswitchon = True
    ),
    TH1StripNoise2ApvCycle = SiStripMonitorCluster.TH1StripNoise2ApvCycle.clone(
        globalswitchon = True
    ),
    TH1StripNoise3ApvCycle = SiStripMonitorCluster.TH1StripNoise3ApvCycle.clone(
        globalswitchon = True
    ),
    ClusterHisto = True,
    BPTXfilter = genericTriggerEventFlag4L1bd
)

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(SiStripMonitorClusterBPTX, 
                         BPTXfilter = dict(
                             stage2 = cms.bool(True),
                             l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                             l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                             ReadPrescalesFromFile = cms.bool(True)
                         ),
                         PixelDCSfilter = dict(
                             stage2 = cms.bool(True),
                             l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                             l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                             ReadPrescalesFromFile = cms.bool(True)),
                         StripDCSfilter = dict(
                             stage2 = cms.bool(True),
                             l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                             l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                             ReadPrescalesFromFile = cms.bool(True)
                         )
                        )

# refitter ### (FIXME rename, move)
from DQM.SiPixelMonitorTrack.RefitterForPixelDQM import *

# Clone for SiStripMonitorTrack for all PDs but Minimum Bias and Jet ####
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrackCommon = SiStripMonitorTrack.clone(
    TrackProducer = 'generalTracks',
    Mod_On = False,
    TH1ClusterCharge = SiStripMonitorTrack.TH1ClusterCharge.clone(
        ringView = True
    ),
    TH1ClusterStoNCorr = SiStripMonitorTrack.TH1ClusterStoNCorr.clone(
        ringView = True
    ),
    TH1ClusterPos = SiStripMonitorTrack.TH1ClusterPos.clone(
        layerView = False,
        ringView = True
    )
)

# Clone for SiStripMonitorTrack for Minimum Bias ####
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrackMB = SiStripMonitorTrack.clone(
    TrackProducer = 'generalTracks',
    Mod_On = False,
    genericTriggerEventPSet = genericTriggerEventFlag4HLTdb,
    TH1ClusterCharge = SiStripMonitorTrack.TH1ClusterCharge.clone(
        ringView = True
    ),
    TH1ClusterStoNCorr = SiStripMonitorTrack.TH1ClusterStoNCorr.clone(
        ringView = True
    )
)

# Clone for SiStripMonitorTrack for Isolated Bunches ####
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrackIB = SiStripMonitorTrack.clone(
    TrackProducer = 'generalTracks',
    Mod_On = False,
    genericTriggerEventPSet = genericTriggerEventFlag4HLTdbIB,
    TH1ClusterCharge = SiStripMonitorTrack.TH1ClusterCharge.clone(
        ringView = True
    ),
    TH1ClusterStoNCorr = SiStripMonitorTrack.TH1ClusterStoNCorr.clone(
        ringView = True
    ),
    TkHistoMap_On = False,
    TH1ClusterNoise = SiStripMonitorTrack.TH1ClusterNoise.clone(
        layerView = False
    ),
    TH1ClusterWidth = SiStripMonitorTrack.TH1ClusterWidth.clone(
        layerView = False
    ),
    TH1ClusterChargePerCM = SiStripMonitorTrack.TH1ClusterChargePerCM.clone(
        ringView = False
    ),
    TopFolderName = "SiStrip/IsolatedBunches"
)

### TrackerMonitorTrack defined and used only for MinimumBias ####
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResiduals.trajectoryInput = 'refittedForPixelDQM'
MonitorTrackResiduals.Tracks          = 'refittedForPixelDQM'
MonitorTrackResiduals.Mod_On        = False
MonitorTrackResiduals.genericTriggerEventPSet = genericTriggerEventFlag4HLTdb

# DQM Services
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoSiStrip = DQMEDAnalyzer('DQMEventInfo',
    subSystemFolder = cms.untracked.string('SiStrip')
)

# Services needed for TkHistoMap
from CalibTracker.SiStripCommon.TkDetMapESProducer_cfi import *

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

from Configuration.ProcessModifiers.approxSiStripClusters_cff import approxSiStripClusters

SiStripDQMTier0 = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorClusterBPTX
    *SiStripMonitorTrackCommon*SiStripMonitorTrackIB*refittedForPixelDQM*MonitorTrackResiduals
    *dqmInfoSiStrip)

from DQM.SiStripMonitorApproximateCluster.SiStripMonitorApproximateCluster_cfi import SiStripMonitorApproximateCluster
SiStripDQMTier0_approx = SiStripDQMTier0.copy()
SiStripDQMTier0_approx += cms.Sequence(SiStripMonitorApproximateCluster)
approxSiStripClusters.toReplaceWith(SiStripDQMTier0, SiStripDQMTier0_approx)

SiStripDQMTier0Common = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorClusterBPTX        
    *SiStripMonitorTrackCommon*SiStripMonitorTrackIB
    *dqmInfoSiStrip)

SiStripDQMTier0Common_approx = SiStripDQMTier0Common.copy()
SiStripDQMTier0Common_approx += cms.Sequence(SiStripMonitorApproximateCluster)
approxSiStripClusters.toReplaceWith(SiStripDQMTier0Common, SiStripDQMTier0Common_approx)

SiStripDQMTier0MinBias = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorClusterBPTX
    *SiStripMonitorTrackMB*SiStripMonitorTrackIB*refittedForPixelDQM*MonitorTrackResiduals
    *dqmInfoSiStrip)

SiStripDQMTier0MinBias_approx = SiStripDQMTier0MinBias.copy()
SiStripDQMTier0MinBias_approx += cms.Sequence(SiStripMonitorApproximateCluster)
approxSiStripClusters.toReplaceWith(SiStripDQMTier0MinBias, SiStripDQMTier0MinBias_approx)
