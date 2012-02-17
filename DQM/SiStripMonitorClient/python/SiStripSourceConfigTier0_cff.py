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



# SiStripMonitorTrack ####
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrack.TrackProducer = 'generalTracks'
SiStripMonitorTrack.Mod_On        = False
SiStripMonitorTrack.andOr         = cms.bool( False )
SiStripMonitorTrack.l1DBKey       = cms.string( 'SiStripDQM_L1' )
SiStripMonitorTrack.l1Algorithms  = cms.vstring(
#                  'L1Tech_BSC_minBias_threshold2.v0 OR L1_BscMinBiasOR_BptxPlusORMinus' # Tech BIT41 OR Algo Bit 124
                   'NOT L1Tech_BSC_halo_beam2_inner.v0' # NOT 36
                 , 'NOT L1Tech_BSC_halo_beam2_outer.v0' # NOT 37
                 , 'NOT L1Tech_BSC_halo_beam1_inner.v0' # NOT 38
                 , 'NOT L1Tech_BSC_halo_beam1_outer.v0' # NOT 39
                 , 'NOT (L1Tech_BSC_splash_beam1.v0 AND NOT L1Tech_BSC_splash_beam2.v0)'   # NOT (42 AND NOT 43)
                 , 'NOT (L1Tech_BSC_splash_beam2.v0 AND NOT L1Tech_BSC_splash_beam1.v0)')  # NOT (43 AND NOT 42)
SiStripMonitorTrack.andOrL1       = cms.bool( False )
SiStripMonitorTrack.errorReplyL1  = cms.bool( True )

# TrackerMonitorTrack ####
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResiduals.trajectoryInput = 'generalTracks'
MonitorTrackResiduals.OutputMEsInRootFile = False
MonitorTrackResiduals.Mod_On        = False
MonitorTrackResiduals.andOr         = cms.bool( False )
MonitorTrackResiduals.l1DBKey       = cms.string( 'SiStripDQM_L1' )
MonitorTrackResiduals.l1Algorithms  = cms.vstring(
#                  'L1Tech_BSC_minBias_threshold2.v0 OR L1_BscMinBiasOR_BptxPlusORMinus' # Tech BIT41 OR Algo Bit 124  
                   'NOT L1Tech_BSC_halo_beam2_inner.v0' # NOT 36
                 , 'NOT L1Tech_BSC_halo_beam2_outer.v0' # NOT 37
                 , 'NOT L1Tech_BSC_halo_beam1_inner.v0' # NOT 38
                 , 'NOT L1Tech_BSC_halo_beam1_outer.v0' # NOT 39
                 , 'NOT (L1Tech_BSC_splash_beam1.v0 AND NOT L1Tech_BSC_splash_beam2.v0)'   # NOT (42 AND NOT 43)
                 , 'NOT (L1Tech_BSC_splash_beam2.v0 AND NOT L1Tech_BSC_splash_beam1.v0)')  # NOT (43 AND NOT 42)
MonitorTrackResiduals.andOrL1       = cms.bool( False )
MonitorTrackResiduals.errorReplyL1  = cms.bool( True )

# TrackingMonitor ####
from DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi import *
TrackerCollisionTrackMon.FolderName          = 'Tracking/TrackParameters'
TrackerCollisionTrackMon.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'
TrackerCollisionTrackMon.andOr         = cms.bool( False )
TrackerCollisionTrackMon.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionTrackMon.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionTrackMon.andOrDcs      = cms.bool( False )
TrackerCollisionTrackMon.errorReplyDcs = cms.bool( True )
TrackerCollisionTrackMon.l1DBKey       = cms.string( 'SiStripDQM_L1' )
TrackerCollisionTrackMon.l1Algorithms  = cms.vstring(
#                  'L1Tech_BSC_minBias_threshold2.v0 OR L1_BscMinBiasOR_BptxPlusORMinus' # Tech BIT41 OR Algo Bit 124     
                   'NOT L1Tech_BSC_halo_beam2_inner.v0' # NOT 36
                 , 'NOT L1Tech_BSC_halo_beam2_outer.v0' # NOT 37
                 , 'NOT L1Tech_BSC_halo_beam1_inner.v0' # NOT 38
                 , 'NOT L1Tech_BSC_halo_beam1_outer.v0' # NOT 39
                 , 'NOT (L1Tech_BSC_splash_beam1.v0 AND NOT L1Tech_BSC_splash_beam2.v0)'   # NOT (42 AND NOT 43)
                 , 'NOT (L1Tech_BSC_splash_beam2.v0 AND NOT L1Tech_BSC_splash_beam1.v0)')  # NOT (43 AND NOT 42)
TrackerCollisionTrackMon.andOrL1       = cms.bool( False )
TrackerCollisionTrackMon.errorReplyL1  = cms.bool( True )

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

# Sequence
SiStripDQMTier0 = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorCluster
    *SiStripMonitorTrack*MonitorTrackResiduals
    *TrackerCollisionTrackMon
    *TrackMonStep0*TrackMonStep1*TrackMonStep2*TrackMonStep3*TrackMonStep4*TrackMonStep5*TrackMonStep6
    *dqmInfoSiStrip)
