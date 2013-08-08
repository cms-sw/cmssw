import FWCore.ParameterSet.Config as cms


# HLT Online -----------------------------------
# AlCa
#from DQM.HLTEvF.HLTAlCaMonPi0_cfi import *
#from DQM.HLTEvF.HLTAlCaMonEcalPhiSym_cfi import *
# JetMET
#from DQM.HLTEvF.HLTMonJetMETDQMSource_cfi import *
# Electron
#from DQM.HLTEvF.HLTMonEleBits_cfi import *
# Muon
#from DQM.HLTEvF.HLTMonMuonDQM_cfi import *
#from DQM.HLTEvF.HLTMonMuonBits_cfi import *
# Photon
#from DQM.HLTEvF.HLTMonPhotonBits_cfi import *
# Tau
#from DQM.HLTEvF.HLTMonTau_cfi import *
#from DQM.HLTEvF.hltMonBTagIPSource_cfi import *
#from DQM.HLTEvF.hltMonBTagMuSource_cfi import *
# hltMonjmDQM  bombs
# hltMonMuDQM dumps names of all histograms in the directory
# hltMonPhotonBits in future releases
# *hltMonJetMET makes a log file, need to learn how to turn it off
# *hltMonEleBits causes SegmentFaults in HARVESTING(step3) in inlcuded in step2

#import DQMServices.Components.DQMEnvironment_cfi
#dqmEnvHLTOnline = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
#dqmEnvHLTOnline.subSystemFolder = 'HLT'

#onlineHLTSource = cms.Sequence(EcalPi0Mon*EcalPhiSymMon*hltMonEleBits*hltMonMuBits*hltMonTauReco*hltMonBTagIPSource*hltMonBTagMuSource*dqmEnvHLTOnline)
#onlineHLTSource = cms.Sequence(EcalPi0Mon*EcalPhiSymMon*hltMonMuBits*dqmEnvHLTOnline)

# HLT Offline -----------------------------------

# FourVector
#from DQMOffline.Trigger.FourVectorHLTOffline_cfi import *
# Egamma
from DQMOffline.Trigger.HLTGeneralOffline_cfi import *

from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *
#from DQMOffline.Trigger.TopElectronHLTOfflineSource_cfi import *
# Muon
from DQMOffline.Trigger.MuonOffline_Trigger_cff import *
# Top
#from DQMOffline.Trigger.QuadJetAna_cfi import *
# Tau
from DQMOffline.Trigger.HLTTauDQMOffline_cff import *
# JetMET
#from DQMOffline.Trigger.JetMETHLTOfflineSource_cfi import *
from DQMOffline.Trigger.JetMETHLTOfflineAnalyzer_cff import *
# TnP
#from DQMOffline.Trigger.TnPEfficiency_cff import *
# Inclusive VBF
from DQMOffline.Trigger.HLTInclusiveVBFSource_cfi import *

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonHLT.FolderName    = 'HLT/Tracking'
TrackerCollisionTrackMonHLT.TrackProducer    = 'hltPixelTracks'

#### TEST track collection
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonHLT.FolderName    = 'HLT/Tracking/iter4'
TrackerCollisionTrackMonHLT.TrackProducer = 'hltIter4Merged'


# SiStripCluster converter
import EventFilter.SiStripRawToDigi.SiStripRawToClustersHLTdsvbuilder_cff
HLTsiStripClusters = EventFilter.SiStripRawToDigi.SiStripRawToClustersHLTdsvbuilder_cff.siStripClusters.clone()
HLTsiStripClusters.SiStripLazyGetter = cms.InputTag("hltSiStripRawToClustersFacility")
HLTsiStripClusters.SiStripRefGetter  = cms.InputTag("hltSiStripClusters")

# SiStripCluster monitoring
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
HLTSiStripMonitorCluster = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()
HLTSiStripMonitorCluster.ClusterProducerStrip = cms.InputTag("HLTsiStripClusters")
HLTSiStripMonitorCluster.ClusterProducerPix   = cms.InputTag("hltSiPixelClusters")
HLTSiStripMonitorCluster.TopFolderName        = cms.string("HLT/SiStrip")
HLTSiStripMonitorCluster.Mod_On = False
HLTSiStripMonitorCluster.TH1TotalNumberOfClusters.subdetswitchon   = True
HLTSiStripMonitorCluster.TProfClustersApvCycle.subdetswitchon      = False
HLTSiStripMonitorCluster.TProfTotalNumberOfClusters.subdetswitchon = True
HLTSiStripMonitorCluster.TH2CStripVsCpixel.globalswitchon       = True
HLTSiStripMonitorCluster.TH1MultiplicityRegions.globalswitchon  = True
HLTSiStripMonitorCluster.TH1MainDiagonalPosition.globalswitchon = True
HLTSiStripMonitorCluster.TH1StripNoise2ApvCycle.globalswitchon  = False
HLTSiStripMonitorCluster.TH1StripNoise3ApvCycle.globalswitchon  = False
HLTSiStripMonitorCluster.ClusterHisto = True
HLTSiStripMonitorCluster.BPTXfilter = cms.PSet(
        andOr         = cms.bool( False ),
            dbLabel       = cms.string("SiStripDQMTrigger"),
            l1Algorithms = cms.vstring( 'L1Tech_BPTX_plus_AND_minus.v0', 'L1_ZeroBias' ),
            andOrL1       = cms.bool( True ),
            errorReplyL1  = cms.bool( True ),
            l1BeforeMask  = cms.bool( True ) # specifies, if the L1 algorithm decision should be read as before (true) or after (false) masking is applied.
        )
HLTSiStripMonitorCluster.PixelDCSfilter = cms.PSet(
        andOr         = cms.bool( False ),
            dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
            dcsPartitions = cms.vint32 ( 28, 29),
            andOrDcs      = cms.bool( False ),
            errorReplyDcs = cms.bool( True ),
        )
HLTSiStripMonitorCluster.StripDCSfilter = cms.PSet(
        andOr         = cms.bool( False ),
            dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
            dcsPartitions = cms.vint32 ( 24, 25, 26, 27 ),
            andOrDcs      = cms.bool( False ),
            errorReplyDcs = cms.bool( True ),
        )

import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
HLTSiStripMonitorTrack = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
HLTSiStripMonitorTrack.TrackProducer     = 'hltIter4Merged'
HLTSiStripMonitorTrack.TrackLabel        = ''
HLTSiStripMonitorTrack.TrajectoryInEvent = cms.bool(False)
HLTSiStripMonitorTrack.AlgoName          = cms.string("HLT")
HLTSiStripMonitorTrack.RawDigis_On       = cms.bool(False)
HLTSiStripMonitorTrack.Cluster_src       = cms.InputTag('HLTsiStripClusters')
HLTSiStripMonitorTrack.Trend_On          = cms.bool(True)
HLTSiStripMonitorTrack.TopFolderName     = cms.string('HLT/SiStrip')

offlineHLTtrackerSource = cms.Sequence(
#    HLTSiPixelDigiSource *
#    HLTSiPixelClusterSource *
    HLTsiStripClusters *
    HLTSiStripMonitorCluster *
    HLTSiStripMonitorTrack
)    


import DQMServices.Components.DQMEnvironment_cfi
dqmEnvHLT= DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
dqmEnvHLT.subSystemFolder = 'HLT'

#offlineHLTSource = cms.Sequence(hltResults*egHLTOffDQMSource*topElectronHLTOffDQMSource*muonFullOfflineDQM*quadJetAna*HLTTauDQMOffline*jetMETHLTOfflineSource*TnPEfficiency*dqmEnvHLT)

# Remove topElectronHLTOffDQMSource
# remove quadJetAna
offlineHLTSource = cms.Sequence(
    hltResults *
    egHLTOffDQMSource *
    muonFullOfflineDQM *
    HLTTauDQMOffline *
    #jetMETHLTOfflineSource *
    jetMETHLTOfflineAnalyzer *
    #TnPEfficiency *
    hltInclusiveVBFSource *
    TrackerCollisionTrackMonHLT *
#### tracker monitor @ HLT
    offlineHLTtrackerSource *
    dqmEnvHLT)

#triggerOfflineDQMSource =  cms.Sequence(onlineHLTSource*offlineHLTSource)
triggerOfflineDQMSource =  cms.Sequence(offlineHLTSource)
 
