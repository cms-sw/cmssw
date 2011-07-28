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

# SiStripMonitorCluster ####
from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *
SiStripMonitorCluster.Mod_On = False
SiStripMonitorCluster.TProfClustersApvCycle.subdetswitchon = True
SiStripMonitorCluster.TProfTotalNumberOfClusters.subdetswitchon = True

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

# Step0
from DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi import *
TrackerCollisionStep0 = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionStep0.TrackProducer = cms.InputTag("zeroStepTracksWithQuality")
TrackerCollisionStep0.SeedProducer  = cms.InputTag("newSeedFromTriplets")
TrackerCollisionStep0.TCProducer    = cms.InputTag("newTrackCandidateMaker")
TrackerCollisionStep0.AlgoName      = cms.string('Step0')
TrackerCollisionStep0.FolderName          = 'Tracking/TrackParameters'
TrackerCollisionStep0.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'
TrackerCollisionStep0.andOr         = cms.bool( False )
TrackerCollisionStep0.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionStep0.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionStep0.andOrDcs      = cms.bool( False )
TrackerCollisionStep0.errorReplyDcs = cms.bool( True )
TrackerCollisionStep0.l1DBKey       = cms.string( 'SiStripDQM_L1' )
TrackerCollisionStep0.l1Algorithms  = cms.vstring(
#                  'L1Tech_BSC_minBias_threshold2.v0 OR L1_BscMinBiasOR_BptxPlusORMinus' # Tech BIT41 OR Algo Bit 124     
                   'NOT L1Tech_BSC_halo_beam2_inner.v0' # NOT 36
                 , 'NOT L1Tech_BSC_halo_beam2_outer.v0' # NOT 37
                 , 'NOT L1Tech_BSC_halo_beam1_inner.v0' # NOT 38
                 , 'NOT L1Tech_BSC_halo_beam1_outer.v0' # NOT 39
                 , 'NOT (L1Tech_BSC_splash_beam1.v0 AND NOT L1Tech_BSC_splash_beam2.v0)'   # NOT (42 AND NOT 43)
                 , 'NOT (L1Tech_BSC_splash_beam2.v0 AND NOT L1Tech_BSC_splash_beam1.v0)')  # NOT (43 AND NOT 42)    
TrackerCollisionStep0.andOrL1       = cms.bool( False )
TrackerCollisionStep0.errorReplyL1  = cms.bool( True )

TrackerCollisionStep0.doGoodTrackPlots   = cms.bool(False)
TrackerCollisionStep0.doTrackerSpecific     = cms.bool(False)
TrackerCollisionStep0.doAllPlots            = cms.bool(False)
TrackerCollisionStep0.doBeamSpotPlots       = cms.bool(False)
TrackerCollisionStep0.doSeedParameterHistos = cms.bool(False)
TrackerCollisionStep0.doLumiAnalysis        = cms.bool(False)
TrackerCollisionStep0.TkSeedSizeBin = cms.int32(50)
TrackerCollisionStep0.TkSeedSizeMax = cms.double(5000)
TrackerCollisionStep0.ClusterLabels = cms.vstring('Pix')

# Step1
from DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi import *
TrackerCollisionStep1 = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionStep1.TrackProducer = cms.InputTag("preMergingFirstStepTracksWithQuality")
TrackerCollisionStep1.SeedProducer  = cms.InputTag("newSeedFromPairs")
TrackerCollisionStep1.TCProducer    = cms.InputTag("stepOneTrackCandidateMaker")
TrackerCollisionStep1.AlgoName      = cms.string('Step1')
TrackerCollisionStep1.FolderName          = 'Tracking/TrackParameters'
TrackerCollisionStep1.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'
TrackerCollisionStep1.andOr         = cms.bool( False )
TrackerCollisionStep1.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionStep1.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionStep1.andOrDcs      = cms.bool( False )
TrackerCollisionStep1.errorReplyDcs = cms.bool( True )
TrackerCollisionStep1.l1DBKey       = cms.string( 'SiStripDQM_L1' )
TrackerCollisionStep1.l1Algorithms  = cms.vstring(
#                  'L1Tech_BSC_minBias_threshold2.v0 OR L1_BscMinBiasOR_BptxPlusORMinus' # Tech BIT41 OR Algo Bit 124     
                   'NOT L1Tech_BSC_halo_beam2_inner.v0' # NOT 36
                 , 'NOT L1Tech_BSC_halo_beam2_outer.v0' # NOT 37
                 , 'NOT L1Tech_BSC_halo_beam1_inner.v0' # NOT 38
                 , 'NOT L1Tech_BSC_halo_beam1_outer.v0' # NOT 39
                 , 'NOT (L1Tech_BSC_splash_beam1.v0 AND NOT L1Tech_BSC_splash_beam2.v0)'   # NOT (42 AND NOT 43)
                 , 'NOT (L1Tech_BSC_splash_beam2.v0 AND NOT L1Tech_BSC_splash_beam1.v0)')  # NOT (43 AND NOT 42)    
TrackerCollisionStep1.andOrL1       = cms.bool( False )
TrackerCollisionStep1.errorReplyL1  = cms.bool( True )

TrackerCollisionStep1.doGoodTrackPlots   = cms.bool(False)
TrackerCollisionStep1.doTrackerSpecific     = cms.bool(False)
TrackerCollisionStep1.doAllPlots            = cms.bool(False)
TrackerCollisionStep1.doBeamSpotPlots       = cms.bool(False)
TrackerCollisionStep1.doSeedParameterHistos = cms.bool(False)
TrackerCollisionStep1.doLumiAnalysis        = cms.bool(False)   
TrackerCollisionStep1.TkSeedSizeBin = cms.int32(500)
TrackerCollisionStep1.TkSeedSizeMax = cms.double(500000)
TrackerCollisionStep1.ClusterLabels = cms.vstring('Pix')

# Step2
from DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi import *
TrackerCollisionStep2 = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionStep2.TrackProducer = cms.InputTag("secStep")
TrackerCollisionStep2.SeedProducer  = cms.InputTag("secTriplets")
TrackerCollisionStep2.TCProducer    = cms.InputTag("secTrackCandidates")
TrackerCollisionStep2.AlgoName      = cms.string('Step2')
TrackerCollisionStep2.FolderName          = 'Tracking/TrackParameters'
TrackerCollisionStep2.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'
TrackerCollisionStep2.andOr         = cms.bool( False )
TrackerCollisionStep2.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionStep2.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionStep2.andOrDcs      = cms.bool( False )
TrackerCollisionStep2.errorReplyDcs = cms.bool( True )
TrackerCollisionStep2.l1DBKey       = cms.string( 'SiStripDQM_L1' )
TrackerCollisionStep2.l1Algorithms  = cms.vstring(
#                  'L1Tech_BSC_minBias_threshold2.v0 OR L1_BscMinBiasOR_BptxPlusORMinus' # Tech BIT41 OR Algo Bit 124     
                   'NOT L1Tech_BSC_halo_beam2_inner.v0' # NOT 36
                 , 'NOT L1Tech_BSC_halo_beam2_outer.v0' # NOT 37
                 , 'NOT L1Tech_BSC_halo_beam1_inner.v0' # NOT 38
                 , 'NOT L1Tech_BSC_halo_beam1_outer.v0' # NOT 39
                 , 'NOT (L1Tech_BSC_splash_beam1.v0 AND NOT L1Tech_BSC_splash_beam2.v0)'   # NOT (42 AND NOT 43)
                 , 'NOT (L1Tech_BSC_splash_beam2.v0 AND NOT L1Tech_BSC_splash_beam1.v0)')  # NOT (43 AND NOT 42)    
TrackerCollisionStep2.andOrL1       = cms.bool( False )
TrackerCollisionStep2.errorReplyL1  = cms.bool( True )

TrackerCollisionStep2.doGoodTrackPlots   = cms.bool(False)
TrackerCollisionStep2.doTrackerSpecific     = cms.bool(False)
TrackerCollisionStep2.doAllPlots            = cms.bool(False)
TrackerCollisionStep2.doBeamSpotPlots       = cms.bool(False)
TrackerCollisionStep2.doSeedParameterHistos = cms.bool(False)
TrackerCollisionStep2.doLumiAnalysis        = cms.bool(False)
TrackerCollisionStep2.TkSeedSizeBin = cms.int32(250)
TrackerCollisionStep2.TkSeedSizeMax = cms.double(50000)
TrackerCollisionStep2.ClusterLabels = cms.vstring('Pix')

# Step3
from DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi import *
TrackerCollisionStep3 = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionStep3.TrackProducer = cms.InputTag("thStep")
TrackerCollisionStep3.SeedProducer  = cms.InputTag("thTriplets")
TrackerCollisionStep3.TCProducer    = cms.InputTag("thTrackCandidates")
TrackerCollisionStep3.AlgoName      = cms.string('Step3')
TrackerCollisionStep3.FolderName          = 'Tracking/TrackParameters'
TrackerCollisionStep3.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'
TrackerCollisionStep3.andOr         = cms.bool( False )
TrackerCollisionStep3.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionStep3.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionStep3.andOrDcs      = cms.bool( False )
TrackerCollisionStep3.errorReplyDcs = cms.bool( True )
TrackerCollisionStep3.l1DBKey       = cms.string( 'SiStripDQM_L1' )
TrackerCollisionStep3.l1Algorithms  = cms.vstring(
#                  'L1Tech_BSC_minBias_threshold2.v0 OR L1_BscMinBiasOR_BptxPlusORMinus' # Tech BIT41 OR Algo Bit 124     
                   'NOT L1Tech_BSC_halo_beam2_inner.v0' # NOT 36
                 , 'NOT L1Tech_BSC_halo_beam2_outer.v0' # NOT 37
                 , 'NOT L1Tech_BSC_halo_beam1_inner.v0' # NOT 38
                 , 'NOT L1Tech_BSC_halo_beam1_outer.v0' # NOT 39
                 , 'NOT (L1Tech_BSC_splash_beam1.v0 AND NOT L1Tech_BSC_splash_beam2.v0)'   # NOT (42 AND NOT 43)
                 , 'NOT (L1Tech_BSC_splash_beam2.v0 AND NOT L1Tech_BSC_splash_beam1.v0)')  # NOT (43 AND NOT 42)    
TrackerCollisionStep3.andOrL1       = cms.bool( False )
TrackerCollisionStep3.errorReplyL1  = cms.bool( True )

TrackerCollisionStep3.doGoodTrackPlots   = cms.bool(False)
TrackerCollisionStep3.doTrackerSpecific     = cms.bool(False)
TrackerCollisionStep3.doAllPlots            = cms.bool(False)
TrackerCollisionStep3.doBeamSpotPlots       = cms.bool(False)
TrackerCollisionStep3.doSeedParameterHistos = cms.bool(False)
TrackerCollisionStep3.doLumiAnalysis        = cms.bool(False)

TrackerCollisionStep3.TkSeedSizeBin = cms.int32(500)
TrackerCollisionStep3.TkSeedSizeMax = cms.double(150000)
TrackerCollisionStep3.ClusterLabels = cms.vstring('Pix')

# Step4
from DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi import *
TrackerCollisionStep4 = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionStep4.TrackProducer = cms.InputTag("pixellessStep")
TrackerCollisionStep4.SeedProducer  = cms.InputTag("fourthPLSeeds")
TrackerCollisionStep4.TCProducer    = cms.InputTag("fourthTrackCandidates")
TrackerCollisionStep4.AlgoName      = cms.string('Step4')
TrackerCollisionStep4.FolderName          = 'Tracking/TrackParameters'
TrackerCollisionStep4.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'
TrackerCollisionStep4.andOr         = cms.bool( False )
TrackerCollisionStep4.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionStep4.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionStep4.andOrDcs      = cms.bool( False )
TrackerCollisionStep4.errorReplyDcs = cms.bool( True )
TrackerCollisionStep4.l1DBKey       = cms.string( 'SiStripDQM_L1' )
TrackerCollisionStep4.l1Algorithms  = cms.vstring(
#                  'L1Tech_BSC_minBias_threshold2.v0 OR L1_BscMinBiasOR_BptxPlusORMinus' # Tech BIT41 OR Algo Bit 124     
                   'NOT L1Tech_BSC_halo_beam2_inner.v0' # NOT 36
                 , 'NOT L1Tech_BSC_halo_beam2_outer.v0' # NOT 37
                 , 'NOT L1Tech_BSC_halo_beam1_inner.v0' # NOT 38
                 , 'NOT L1Tech_BSC_halo_beam1_outer.v0' # NOT 39
                 , 'NOT (L1Tech_BSC_splash_beam1.v0 AND NOT L1Tech_BSC_splash_beam2.v0)'   # NOT (42 AND NOT 43)
                 , 'NOT (L1Tech_BSC_splash_beam2.v0 AND NOT L1Tech_BSC_splash_beam1.v0)')  # NOT (43 AND NOT 42)    
TrackerCollisionStep4.andOrL1       = cms.bool( False )
TrackerCollisionStep4.errorReplyL1  = cms.bool( True )

TrackerCollisionStep4.doGoodTrackPlots   = cms.bool(False)
TrackerCollisionStep4.doTrackerSpecific     = cms.bool(False)
TrackerCollisionStep4.doAllPlots            = cms.bool(False)
TrackerCollisionStep4.doBeamSpotPlots       = cms.bool(False)
TrackerCollisionStep4.doSeedParameterHistos = cms.bool(False)
TrackerCollisionStep4.doLumiAnalysis        = cms.bool(False)   

TrackerCollisionStep4.TkSeedSizeBin = cms.int32(250)
TrackerCollisionStep4.TkSeedSizeMax = cms.double(50000)
TrackerCollisionStep4.ClusterLabels = cms.vstring('Strip')

# Step5
from DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi import *
TrackerCollisionStep5 = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionStep5.TrackProducer = cms.InputTag("tobtecStep")
TrackerCollisionStep5.SeedProducer  = cms.InputTag("fifthSeeds")
TrackerCollisionStep5.TCProducer    = cms.InputTag("fifthTrackCandidates")
TrackerCollisionStep5.AlgoName      = cms.string('Step5')
TrackerCollisionStep5.FolderName          = 'Tracking/TrackParameters'
TrackerCollisionStep5.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'
TrackerCollisionStep5.andOr         = cms.bool( False )
TrackerCollisionStep5.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackerCollisionStep5.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackerCollisionStep5.andOrDcs      = cms.bool( False )
TrackerCollisionStep5.errorReplyDcs = cms.bool( True )
TrackerCollisionStep5.l1DBKey       = cms.string( 'SiStripDQM_L1' )
TrackerCollisionStep5.l1Algorithms  = cms.vstring(
#                  'L1Tech_BSC_minBias_threshold2.v0 OR L1_BscMinBiasOR_BptxPlusORMinus' # Tech BIT41 OR Algo Bit 124     
                   'NOT L1Tech_BSC_halo_beam2_inner.v0' # NOT 36
                 , 'NOT L1Tech_BSC_halo_beam2_outer.v0' # NOT 37
                 , 'NOT L1Tech_BSC_halo_beam1_inner.v0' # NOT 38
                 , 'NOT L1Tech_BSC_halo_beam1_outer.v0' # NOT 39
                 , 'NOT (L1Tech_BSC_splash_beam1.v0 AND NOT L1Tech_BSC_splash_beam2.v0)'   # NOT (42 AND NOT 43)
                 , 'NOT (L1Tech_BSC_splash_beam2.v0 AND NOT L1Tech_BSC_splash_beam1.v0)')  # NOT (43 AND NOT 42)    
TrackerCollisionStep5.andOrL1       = cms.bool( False )
TrackerCollisionStep5.errorReplyL1  = cms.bool( True )

TrackerCollisionStep5.doGoodTrackPlots   = cms.bool(False)
TrackerCollisionStep5.doTrackerSpecific     = cms.bool(False)
TrackerCollisionStep5.doAllPlots            = cms.bool(False)
TrackerCollisionStep5.doBeamSpotPlots       = cms.bool(False)
TrackerCollisionStep5.doSeedParameterHistos = cms.bool(False)
TrackerCollisionStep5.doLumiAnalysis        = cms.bool(False)   

TrackerCollisionStep5.TkSeedSizeBin = cms.int32(250)
TrackerCollisionStep5.TkSeedSizeMax = cms.double(50000)
TrackerCollisionStep5.ClusterLabels = cms.vstring('Strip')

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
    *TrackerCollisionTrackMon*TrackerCollisionStep0*TrackerCollisionStep1*TrackerCollisionStep2
    *TrackerCollisionStep3*TrackerCollisionStep4*TrackerCollisionStep5
    *dqmInfoSiStrip)
