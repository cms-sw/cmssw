import FWCore.ParameterSet.Config as cms

## SiStripCluster converter
#import EventFilter.SiStripRawToDigi.SiStripRawToClustersHLTdsvbuilder_cff
#HLTsiStripClusters = EventFilter.SiStripRawToDigi.SiStripRawToClustersHLTdsvbuilder_cff.siStripClusters.clone()
#HLTsiStripClusters.SiStripLazyGetter = cms.InputTag("hltSiStripRawToClustersFacility")
#HLTsiStripClusters.SiStripRefGetter  = cms.InputTag("hltSiStripClusters")

# SiStripCluster monitoring
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
HLTSiStripMonitorCluster = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()
#HLTSiStripMonitorCluster.ClusterProducerStrip = cms.InputTag("HLTsiStripClusters")
HLTSiStripMonitorCluster.ClusterProducerStrip = cms.InputTag("hltSiStripRawToClustersFacility")
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

from RecoTracker.TrackProducer.TrackRefitter_cfi import *
hltTrackRefitterForSiStripMonitorTrack = TrackRefitter.clone()
hltTrackRefitterForSiStripMonitorTrack.src = cms.InputTag("hltIter4Merged") ## give problem : BadRefCore RefCore: Request to resolve a null or invalid reference to a product of type 'edmNew::DetSetVector<SiStripCluster>' has been detected
#hltTrackRefitterForSiStripMonitorTrack.src = cms.InputTag("hltPFJetCtfWithMaterialTracks") ## give problem : BadRefCore RefCore: Request to resolve a null or invalid reference to a product of type 'edmNew::DetSetVector<SiStripCluster>' has been detected
#hltTrackRefitterForSiStripMonitorTrack.src = cms.InputTag('hltIter4PFJetCkfTrackCandidates') ## give problems w/ trajectory maker
#hltTrackRefitterForSiStripMonitorTrack.src = cms.InputTag('hltIter4PFlowTrackSelectionHighPurity')
#hltTrackRefitterForSiStripMonitorTrack.src = cms.InputTag('hltIter4PFJetCtfWithMaterialTracks')
hltTrackRefitterForSiStripMonitorTrack.TrajectoryInEvent = True

import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
HLTSiStripMonitorTrack = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
#HLTSiStripMonitorTrack.TrackProducer     = 'hltIter4Merged' ## works, but there is not the trajectory map :(
HLTSiStripMonitorTrack.TrackProducer     = 'hltTrackRefitterForSiStripMonitorTrack' 
HLTSiStripMonitorTrack.TrackLabel        = ''
HLTSiStripMonitorTrack.TrajectoryInEvent = cms.bool(True) ### ?!?!?!?!?
HLTSiStripMonitorTrack.AlgoName          = cms.string("HLT")
HLTSiStripMonitorTrack.RawDigis_On       = cms.bool(False) ### ?!?!?!?!?
HLTSiStripMonitorTrack.Cluster_src       = cms.InputTag('hltSiStripRawToClustersFacility')
HLTSiStripMonitorTrack.Trend_On          = cms.bool(True)
HLTSiStripMonitorTrack.TopFolderName     = cms.string('HLT/SiStrip')

sistripMonitorHLTsequence = cms.Sequence(
#    HLTsiStripClusters
#    * HLTSiStripMonitorCluster
    HLTSiStripMonitorCluster
### commented as book keeping, because there is not the possibility of having the trajectory for S/N !!!
#    * hltTrackRefitterForSiStripMonitorTrack
#    * HLTSiStripMonitorTrack
#    hltTrackRefitterForSiStripMonitorTrack
#    * HLTSiStripMonitorTrack
)    
