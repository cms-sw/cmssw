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
hltTrackRefitterForSiStripMonitorTrack.beamSpot                = cms.InputTag("hltOnlineBeamSpot")
hltTrackRefitterForSiStripMonitorTrack.MeasurementTrackerEvent = cms.InputTag('MeasurementTrackerEvent')
hltTrackRefitterForSiStripMonitorTrack.TrajectoryInEvent       = cms.bool(True)
hltTrackRefitterForSiStripMonitorTrack.useHitsSplitting        = cms.bool(False)
#hltTrackRefitterForSiStripMonitorTrack.src                     = cms.InputTag("hltIter4Merged") # scenario 0
hltTrackRefitterForSiStripMonitorTrack.src                     = cms.InputTag("hltIter2Merged") # scenario 1

import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
HLTSiStripMonitorTrack = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
HLTSiStripMonitorTrack.TrackProducer     = 'hltTrackRefitterForSiStripMonitorTrack' 
HLTSiStripMonitorTrack.TrackLabel        = ''
HLTSiStripMonitorTrack.AlgoName          = cms.string("HLT")
HLTSiStripMonitorTrack.Cluster_src       = cms.InputTag('hltSiStripRawToClustersFacility')
HLTSiStripMonitorTrack.Trend_On          = cms.bool(True)
HLTSiStripMonitorTrack.TopFolderName     = cms.string('HLT/SiStrip')
HLTSiStripMonitorTrack.Mod_On            = cms.bool(False)

sistripMonitorHLTsequence = cms.Sequence(
    HLTSiStripMonitorCluster
    * hltTrackRefitterForSiStripMonitorTrack
    * HLTSiStripMonitorTrack
)    

