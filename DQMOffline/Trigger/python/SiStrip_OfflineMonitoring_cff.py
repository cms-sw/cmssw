import FWCore.ParameterSet.Config as cms

## SiStripCluster converter
#import EventFilter.SiStripRawToDigi.SiStripRawToClustersHLTdsvbuilder_cff
#HLTsiStripClusters = EventFilter.SiStripRawToDigi.SiStripRawToClustersHLTdsvbuilder_cff.siStripClusters.clone()
#HLTsiStripClusters.SiStripLazyGetter = cms.InputTag("hltSiStripRawToClustersFacility")
#HLTsiStripClusters.SiStripRefGetter  = cms.InputTag("hltSiStripClusters")

# SiStripCluster monitoring
from RecoLocalTracker.SiPixelRecHits.SiPixelTemplateStoreESProducer_cfi import SiPixelTemplateStoreESProducer
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
HLTSiStripMonitorCluster = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone(
    ClusterProducerStrip = "hltSiStripRawToClustersFacility",
    ClusterProducerPix   = "hltSiPixelClusters",
    TopFolderName        = "HLT/SiStrip",
    ClusterHisto         = True,
    Mod_On               = False
)
HLTSiStripMonitorCluster.TH1TotalNumberOfClusters.subdetswitchon   = cms.bool(True)
HLTSiStripMonitorCluster.TProfClustersApvCycle.subdetswitchon      = cms.bool(False)
HLTSiStripMonitorCluster.TProfTotalNumberOfClusters.subdetswitchon = cms.bool(True)
HLTSiStripMonitorCluster.TH2CStripVsCpixel.globalswitchon       = cms.bool(True)
HLTSiStripMonitorCluster.TH1MultiplicityRegions.globalswitchon  = cms.bool(True)
HLTSiStripMonitorCluster.TH1MainDiagonalPosition.globalswitchon = cms.bool(True)
HLTSiStripMonitorCluster.TH1StripNoise2ApvCycle.globalswitchon  = cms.bool(False)
HLTSiStripMonitorCluster.TH1StripNoise3ApvCycle.globalswitchon  = cms.bool(False)

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
            dcsRecordInputTag = cms.InputTag("onlineMetaDataDigis"),
            dcsPartitions = cms.vint32 ( 28, 29),
            andOrDcs      = cms.bool( False ),
            errorReplyDcs = cms.bool( True ),
        )
HLTSiStripMonitorCluster.StripDCSfilter = cms.PSet(
        andOr         = cms.bool( False ),
            dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
            dcsRecordInputTag = cms.InputTag("onlineMetaDataDigis"),
            dcsPartitions = cms.vint32 ( 24, 25, 26, 27 ),
            andOrDcs      = cms.bool( False ),
            errorReplyDcs = cms.bool( True ),
        )
HLTSiStripMonitorCluster.TH2CStripVsCpixel = cms.PSet(
        Nbinsx = cms.int32(200),
        xmin   = cms.double(-0.5),
        xmax   = cms.double(99999.5),
        Nbinsy = cms.int32(170),
        ymin   = cms.double(-0.5),
        ymax   = cms.double(50999.5),
        globalswitchon = cms.bool(True)
        )
HLTSiStripMonitorCluster.TH1NClusPx = cms.PSet(
        Nbinsx = cms.int32(170),
        xmax = cms.double(50999.5),
        xmin = cms.double(-0.5)
        )
HLTSiStripMonitorCluster.TH1NClusStrip = cms.PSet(
        Nbinsx = cms.int32(200),
        xmax = cms.double(99999.5),
        xmin = cms.double(-0.5)
    )
hltESPPixelCPETemplateReco = cms.ESProducer( "PixelCPETemplateRecoESProducer",
  LoadTemplatesFromDB = cms.bool( True ),
  ComponentName = cms.string( "hltESPPixelCPETemplateReco" ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  speed = cms.int32( -2 ),
  UseClusterSplitter = cms.bool( False )
)

hltESPPixelCPEGeneric = cms.ESProducer( "PixelCPEGenericESProducer",
  EdgeClusterErrorX = cms.double( 50.0 ),
  DoCosmics = cms.bool( False ),
  LoadTemplatesFromDB = cms.bool( True ),
  UseErrorsFromTemplates = cms.bool( True ),
  eff_charge_cut_highX = cms.double( 1.0 ),
  TruncatePixelCharge = cms.bool( True ),
  size_cutY = cms.double( 3.0 ),
  size_cutX = cms.double( 3.0 ),
  inflate_all_errors_no_trk_angle = cms.bool( False ),
  IrradiationBiasCorrection = cms.bool( False ),
  inflate_errors = cms.bool( False ),
  eff_charge_cut_lowX = cms.double( 0.0 ),
  eff_charge_cut_highY = cms.double( 1.0 ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  EdgeClusterErrorY = cms.double( 85.0 ),
  ComponentName = cms.string( "hltESPPixelCPEGeneric" ),
  eff_charge_cut_lowY = cms.double( 0.0 ),
  Alpha2Order = cms.bool( True )
)

hltESPTTRHBuilderAngleAndTemplate = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPETemplateReco" ),
  ComponentName = cms.string( "hltESPTTRHBuilderAngleAndTemplate" )
)

hltESPTTRHBWithTrackAngle = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBWithTrackAngle" )
)

hltESPStripCPEfromTrackAngle = cms.ESProducer( "StripCPEESProducer",
  ComponentType = cms.string( "StripCPEfromTrackAngle" ),
  ComponentName = cms.string( "hltESPStripCPEfromTrackAngle" ),
  parameters = cms.PSet( 
    mLC_P2 = cms.double( 0.3 ),
    mLC_P1 = cms.double( 0.618 ),
    mLC_P0 = cms.double( -0.326 ),
    useLegacyError = cms.bool( True ),
    mTEC_P1 = cms.double( 0.471 ),
    mTEC_P0 = cms.double( -1.885 ),
    mTOB_P0 = cms.double( -1.026 ),
    mTOB_P1 = cms.double( 0.253 ),
    mTIB_P0 = cms.double( -0.742 ),
    mTIB_P1 = cms.double( 0.202 ),
    mTID_P0 = cms.double( -1.427 ),
    mTID_P1 = cms.double( 0.433 )
  )
)

from RecoTracker.TrackProducer.TrackRefitter_cfi import *
hltTrackRefitterForSiStripMonitorTrack = TrackRefitter.clone(
    beamSpot                = "hltOnlineBeamSpot",
    MeasurementTrackerEvent = 'MeasurementTrackerEvent',
    TrajectoryInEvent       = True,
    useHitsSplitting        = False,
    #src                     = "hltIter4Merged", # scenario 0
    src                     = "hltMergedTracks", # hltIter2Merged # scenario 1
    #TTRHBuilder             = 'hltESPTTRHBuilderAngleAndTemplate',
    TTRHBuilder             = 'hltESPTTRHBWithTrackAngle'
)
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
HLTSiStripMonitorTrack = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone(
    TrackProducer     = 'hltTrackRefitterForSiStripMonitorTrack',
    TrackLabel        = '',
    AlgoName          = "HLT",
    Cluster_src       = 'hltSiStripRawToClustersFacility',
    Trend_On          = True,
    TopFolderName     = 'HLT/SiStrip',
    Mod_On            = False
)
sistripMonitorHLTsequence = cms.Sequence(
    HLTSiStripMonitorCluster
    * hltTrackRefitterForSiStripMonitorTrack
    * HLTSiStripMonitorTrack
)    

