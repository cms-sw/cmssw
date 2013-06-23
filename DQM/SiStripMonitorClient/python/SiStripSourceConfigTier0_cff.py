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
SiStripMonitorClusterBPTX = SiStripMonitorCluster.clone()
SiStripMonitorClusterBPTX.Mod_On = False
SiStripMonitorClusterBPTX.TH1TotalNumberOfClusters.subdetswitchon   = True
SiStripMonitorClusterBPTX.TProfClustersApvCycle.subdetswitchon      = True
SiStripMonitorClusterBPTX.TProfTotalNumberOfClusters.subdetswitchon = True
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
### LocalReco
# Clone for all PDs but MinBias ####
LocalRecoLogMessageMonCommon = DQM.TrackingMonitor.LogMessageMonitor_cff.LocalRecoLogMessageMon.clone()
LocalRecoLogMessageMonCommon.andOr         = cms.bool( False )
LocalRecoLogMessageMonCommon.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
LocalRecoLogMessageMonCommon.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
LocalRecoLogMessageMonCommon.andOrDcs      = cms.bool( False )
LocalRecoLogMessageMonCommon.errorReplyDcs = cms.bool( True )

# Clone for MinBias ###
LocalRecoLogMessageMonMB = DQM.TrackingMonitor.LogMessageMonitor_cff.LocalRecoLogMessageMon.clone()
LocalRecoLogMessageMonMB.andOr         = cms.bool( False )
LocalRecoLogMessageMonMB.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
LocalRecoLogMessageMonMB.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
LocalRecoLogMessageMonMB.andOrDcs      = cms.bool( False )
LocalRecoLogMessageMonMB.errorReplyDcs = cms.bool( True )
LocalRecoLogMessageMonMB.dbLabel       = cms.string("SiStripDQMTrigger")
LocalRecoLogMessageMonMB.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
LocalRecoLogMessageMonMB.hltPaths      = cms.vstring("HLT_ZeroBias_*")
LocalRecoLogMessageMonMB.hltDBKey      = cms.string("Tracker_MB")
LocalRecoLogMessageMonMB.errorReplyHlt = cms.bool( False )
LocalRecoLogMessageMonMB.andOrHlt      = cms.bool(True) 

### Clusterizer
# Clone for all PDs but MinBias ####
ClusterizerLogMessageMonCommon = DQM.TrackingMonitor.LogMessageMonitor_cff.ClusterizerLogMessageMon.clone()
ClusterizerLogMessageMonCommon.andOr         = cms.bool( False )
ClusterizerLogMessageMonCommon.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
ClusterizerLogMessageMonCommon.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
ClusterizerLogMessageMonCommon.andOrDcs      = cms.bool( False )
ClusterizerLogMessageMonCommon.errorReplyDcs = cms.bool( True )

# Clone for MinBias ###
ClusterizerLogMessageMonMB = DQM.TrackingMonitor.LogMessageMonitor_cff.ClusterizerLogMessageMon.clone()
ClusterizerLogMessageMonMB.andOr         = cms.bool( False )
ClusterizerLogMessageMonMB.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
ClusterizerLogMessageMonMB.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
ClusterizerLogMessageMonMB.andOrDcs      = cms.bool( False )
ClusterizerLogMessageMonMB.errorReplyDcs = cms.bool( True )
ClusterizerLogMessageMonMB.dbLabel       = cms.string("SiStripDQMTrigger")
ClusterizerLogMessageMonMB.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
ClusterizerLogMessageMonMB.hltPaths      = cms.vstring("HLT_ZeroBias_*")
ClusterizerLogMessageMonMB.hltDBKey      = cms.string("Tracker_MB")
ClusterizerLogMessageMonMB.errorReplyHlt = cms.bool( False )
ClusterizerLogMessageMonMB.andOrHlt      = cms.bool(True) 

### Seeding
# Clone for all PDs but MinBias ####
SeedingLogMessageMonCommon = DQM.TrackingMonitor.LogMessageMonitor_cff.SeedingLogMessageMon.clone()
SeedingLogMessageMonCommon.andOr         = cms.bool( False )
SeedingLogMessageMonCommon.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
SeedingLogMessageMonCommon.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
SeedingLogMessageMonCommon.andOrDcs      = cms.bool( False )
SeedingLogMessageMonCommon.errorReplyDcs = cms.bool( True )

# Clone for MinBias ###
SeedingLogMessageMonMB = DQM.TrackingMonitor.LogMessageMonitor_cff.SeedingLogMessageMon.clone()
SeedingLogMessageMonMB.andOr         = cms.bool( False )
SeedingLogMessageMonMB.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
SeedingLogMessageMonMB.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
SeedingLogMessageMonMB.andOrDcs      = cms.bool( False )
SeedingLogMessageMonMB.errorReplyDcs = cms.bool( True )
SeedingLogMessageMonMB.dbLabel       = cms.string("SiStripDQMTrigger")
SeedingLogMessageMonMB.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
SeedingLogMessageMonMB.hltPaths      = cms.vstring("HLT_ZeroBias_*")
SeedingLogMessageMonMB.hltDBKey      = cms.string("Tracker_MB")
SeedingLogMessageMonMB.errorReplyHlt = cms.bool( False )
SeedingLogMessageMonMB.andOrHlt      = cms.bool(True) 

### TrackCandidate
# Clone for all PDs but MinBias ####
TrackCandidateLogMessageMonCommon = DQM.TrackingMonitor.LogMessageMonitor_cff.TrackCandidateLogMessageMon.clone()
TrackCandidateLogMessageMonCommon.andOr         = cms.bool( False )
TrackCandidateLogMessageMonCommon.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackCandidateLogMessageMonCommon.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackCandidateLogMessageMonCommon.andOrDcs      = cms.bool( False )
TrackCandidateLogMessageMonCommon.errorReplyDcs = cms.bool( True )

# Clone for MinBias ###
TrackCandidateLogMessageMonMB = DQM.TrackingMonitor.LogMessageMonitor_cff.TrackCandidateLogMessageMon.clone()
TrackCandidateLogMessageMonMB.andOr         = cms.bool( False )
TrackCandidateLogMessageMonMB.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackCandidateLogMessageMonMB.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackCandidateLogMessageMonMB.andOrDcs      = cms.bool( False )
TrackCandidateLogMessageMonMB.errorReplyDcs = cms.bool( True )
TrackCandidateLogMessageMonMB.dbLabel       = cms.string("SiStripDQMTrigger")
TrackCandidateLogMessageMonMB.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
TrackCandidateLogMessageMonMB.hltPaths      = cms.vstring("HLT_ZeroBias_*")
TrackCandidateLogMessageMonMB.hltDBKey      = cms.string("Tracker_MB")
TrackCandidateLogMessageMonMB.errorReplyHlt = cms.bool( False )
TrackCandidateLogMessageMonMB.andOrHlt      = cms.bool(True) 

### TrackFinder
# Clone for all PDs but MinBias ####
TrackFinderLogMessageMonCommon = DQM.TrackingMonitor.LogMessageMonitor_cff.TrackFinderLogMessageMon.clone()
TrackFinderLogMessageMonCommon.andOr         = cms.bool( False )
TrackFinderLogMessageMonCommon.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackFinderLogMessageMonCommon.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackFinderLogMessageMonCommon.andOrDcs      = cms.bool( False )
TrackFinderLogMessageMonCommon.errorReplyDcs = cms.bool( True )

# Clone for MinBias ###
TrackFinderLogMessageMonMB = DQM.TrackingMonitor.LogMessageMonitor_cff.TrackFinderLogMessageMon.clone()
TrackFinderLogMessageMonMB.andOr         = cms.bool( False )
TrackFinderLogMessageMonMB.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
TrackFinderLogMessageMonMB.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29)
TrackFinderLogMessageMonMB.andOrDcs      = cms.bool( False )
TrackFinderLogMessageMonMB.errorReplyDcs = cms.bool( True )
TrackFinderLogMessageMonMB.dbLabel       = cms.string("SiStripDQMTrigger")
TrackFinderLogMessageMonMB.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
TrackFinderLogMessageMonMB.hltPaths      = cms.vstring("HLT_ZeroBias_*")
TrackFinderLogMessageMonMB.hltDBKey      = cms.string("Tracker_MB")
TrackFinderLogMessageMonMB.errorReplyHlt = cms.bool( False )
TrackFinderLogMessageMonMB.andOrHlt      = cms.bool(True) 

# dEdx monitor ####
from DQM.TrackingMonitor.dEdxAnalyzer_cff import *
import DQM.TrackingMonitor.dEdxAnalyzer_cfi
# Clone for all PDs but MinBias ####
dEdxMonCommon = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxAnalyzer.clone()

# Clone for MinBias ####
dEdxMonMB = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxAnalyzer.clone()
dEdxMonMB.dEdxParameters.andOr         = cms.bool( False )
dEdxMonMB.dEdxParameters.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
dEdxMonMB.dEdxParameters.hltPaths      = cms.vstring("HLT_ZeroBias_*")
dEdxMonMB.dEdxParameters.hltDBKey      = cms.string("Tracker_MB")
dEdxMonMB.dEdxParameters.errorReplyHlt = cms.bool( False )
dEdxMonMB.dEdxParameters.andOrHlt      = cms.bool(True) 

# Clone for SingleMu ####
dEdxMonMU = DQM.TrackingMonitor.dEdxAnalyzer_cfi.dEdxAnalyzer.clone()
dEdxMonMU.dEdxParameters.andOr         = cms.bool( False )
dEdxMonMU.dEdxParameters.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
dEdxMonMU.dEdxParameters.hltPaths      = cms.vstring("HLT_SingleMu40_Eta2p1_*")
dEdxMonMU.dEdxParameters.errorReplyHlt = cms.bool( False )
dEdxMonMU.dEdxParameters.andOrHlt      = cms.bool(True) 


# temporary patch in order to have BXlumi 
from RecoLuminosity.LumiProducer.lumiProducer_cff import *

# temporary test in order to temporary produce the "goodPrimaryVertexCollection"
# define with a new name if changes are necessary, otherwise simply include
# it from CommonTools/ParticleFlow/python/goodOfflinePrimaryVertices_cfi.py
# uncomment when necessary
from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector
from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices
trackingDQMgoodOfflinePrimaryVertices = goodOfflinePrimaryVertices.clone()
trackingDQMgoodOfflinePrimaryVertices.filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) )
trackingDQMgoodOfflinePrimaryVertices.src=cms.InputTag('offlinePrimaryVertices')
trackingDQMgoodOfflinePrimaryVertices.filter = cms.bool(False)

# Sequence
SiStripDQMTier0 = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorClusterBPTX
    *SiStripMonitorTrackCommon*MonitorTrackResiduals
    # dEdx monitoring
#    *RefitterForDedxDQMDeDx * dedxDQMHarm2SP * dedxDQMHarm2SO * dedxDQMHarm2PO * dEdxMonCommon
     * dedxDQMHarm2SP * dedxDQMHarm2SO * dedxDQMHarm2PO * dEdxMonCommon    

#    # temporary patch in order to have BXlumi
#    * lumiProducer
    # temporary test in order to have the "goodPrimaryVertexCollection"
#    * trackingDQMgoodOfflinePrimaryVertices
    *TrackerCollisionTrackMonCommon
    *TrackMonStep0*TrackMonStep1*TrackMonStep2*TrackMonStep3*TrackMonStep4*TrackMonStep5*TrackMonStep6*TrackMonStep9*TrackMonStep10
     # MessageLog
    * LocalRecoLogMessageMonCommon * ClusterizerLogMessageMonCommon * SeedingLogMessageMonCommon * TrackCandidateLogMessageMonCommon * TrackFinderLogMessageMonCommon
    *dqmInfoSiStrip)

SiStripDQMTier0Common = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorClusterBPTX        
    *SiStripMonitorTrackCommon
    # dEdx monitoring
#    *RefitterForDedxDQMDeDx * dedxDQMHarm2SP * dedxDQMHarm2SO * dedxDQMHarm2PO * dEdxMonCommon
     * dedxDQMHarm2SP * dedxDQMHarm2SO * dedxDQMHarm2PO * dEdxMonCommon    

#    # temporary patch in order to have BXlumi
#    * lumiProducer
#    # temporary test in order to have the "goodPrimaryVertexCollection"
#    * trackingDQMgoodOfflinePrimaryVertices
    *TrackerCollisionTrackMonCommon
    *TrackMonStep0*TrackMonStep1*TrackMonStep2*TrackMonStep3*TrackMonStep4*TrackMonStep5*TrackMonStep6*TrackMonStep9*TrackMonStep10
    # MessageLog
    * LocalRecoLogMessageMonCommon * ClusterizerLogMessageMonCommon * SeedingLogMessageMonCommon * TrackCandidateLogMessageMonCommon * TrackFinderLogMessageMonCommon
    *dqmInfoSiStrip)

SiStripDQMTier0MinBias = cms.Sequence(
    APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorClusterBPTX
    *SiStripMonitorTrackMB*MonitorTrackResiduals
    # dEdx monitoring
#    *RefitterForDedxDQMDeDx * dedxDQMHarm2SP * dedxDQMHarm2SO * dedxDQMHarm2PO * dEdxMonMB
     * dedxDQMHarm2SP * dedxDQMHarm2SO * dedxDQMHarm2PO * dEdxMonMB    

#    * lumiProducer
#    # temporary test in order to have the "goodPrimaryVertexCollection"
#    * trackingDQMgoodOfflinePrimaryVertices
    *TrackerCollisionTrackMonMB
    *TrackMonStep0*TrackMonStep1*TrackMonStep2*TrackMonStep3*TrackMonStep4*TrackMonStep5*TrackMonStep6*TrackMonStep9*TrackMonStep10
    # MessageLog
    * LocalRecoLogMessageMonMB * ClusterizerLogMessageMonMB * SeedingLogMessageMonMB * TrackCandidateLogMessageMonMB * TrackFinderLogMessageMonMB
    *dqmInfoSiStrip)

