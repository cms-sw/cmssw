import FWCore.ParameterSet.Config as cms

# import p+p collision sequences
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_cff import *

SiStripMonitorCluster.Mod_On = False
SiStripMonitorCluster.TH1TotalNumberOfClusters.subdetswitchon   = True
SiStripMonitorCluster.TProfClustersApvCycle.subdetswitchon      = True
SiStripMonitorCluster.TProfTotalNumberOfClusters.subdetswitchon = True
SiStripMonitorCluster.TrendVsLS = True
SiStripMonitorCluster.TH2CStripVsCpixel.globalswitchon       = True
SiStripMonitorCluster.TH1MultiplicityRegions.globalswitchon  = True
SiStripMonitorCluster.TH1MainDiagonalPosition.globalswitchon = True
SiStripMonitorCluster.TH1StripNoise2ApvCycle.globalswitchon  = True
SiStripMonitorCluster.TH1StripNoise3ApvCycle.globalswitchon  = True
SiStripMonitorCluster.ClusterHisto = True

SiStripMonitorCluster.PixelDCSfilter = cms.PSet(
    andOr         = cms.bool( False ),
    dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
    dcsPartitions = cms.vint32 ( 28, 29),
    andOrDcs      = cms.bool( False ),
    errorReplyDcs = cms.bool( True ),
)
SiStripMonitorCluster.StripDCSfilter = cms.PSet(
    andOr         = cms.bool( False ),
    dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
    dcsPartitions = cms.vint32 ( 24, 25, 26, 27 ),
    andOrDcs      = cms.bool( False ),
    errorReplyDcs = cms.bool( True ),
)

# SiStripMonitorTrack ####
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_hi  = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_hi.TrackProducer = "hiGeneralTracks"
SiStripMonitorTrack_hi.Mod_On        = False

# TrackerMonitorTrack ####
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_hi = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_hi.Tracks              = 'hiGeneralTracks'
MonitorTrackResiduals_hi.trajectoryInput     = "hiGeneralTracks"
MonitorTrackResiduals_hi.Mod_On              = False



SiStripDQMTier0_hi = cms.Sequence(APVPhases * consecutiveHEs *
                                  siStripFEDCheck * siStripFEDMonitor *
                                  SiStripMonitorDigi * SiStripMonitorCluster *
                                  SiStripMonitorTrack_hi *
                                  MonitorTrackResiduals_hi)
