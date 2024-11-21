import FWCore.ParameterSet.Config as cms

from TrackingTools.RecoGeometry.RecoGeometries_cff import *
hltESPDummyDetLayerGeometry = DummyDetLayerGeometry.clone(
    ComponentName =  "hltESPDummyDetLayerGeometry" 
)

# strip cluster monitor
# in order to get the crossing angle through the sensor
# the track trajectory is needed
# => the track re-fit has to be performed
# => some ESProducer have to be defined

## NB: the following ESProducer should be the same used in the HLT menu
##     make sure they are not already defined somewhereelse in the final configuration
from RecoLocalTracker.SiPixelRecHits.PixelCPETemplateReco_cfi import templates
hltESPPixelCPETemplateReco = templates.clone(
  LoadTemplatesFromDB =  True ,
  ComponentName = "hltESPPixelCPETemplateReco" ,
  Alpha2Order =  True ,
  ClusterProbComputationFlag =  0 ,
  speed =  -2 ,
  UseClusterSplitter =  False 
)

from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import PixelCPEGenericESProducer
hltESPPixelCPEGeneric = PixelCPEGenericESProducer.clone(
  EdgeClusterErrorX =  50.0 ,
  DoCosmics =  False ,
  LoadTemplatesFromDB =  True ,
  UseErrorsFromTemplates =  True ,
  eff_charge_cut_highX =  1.0 ,
  TruncatePixelCharge =  True ,
  size_cutY =  3.0 ,
  size_cutX =  3.0 ,
  inflate_all_errors_no_trk_angle =  False ,
  IrradiationBiasCorrection =  False ,
  inflate_errors = False ,
  eff_charge_cut_lowX =  0.0 ,
  eff_charge_cut_highY =  1.0 ,
  ClusterProbComputationFlag = 0 ,
  EdgeClusterErrorY = 85.0 ,
  ComponentName = "hltESPPixelCPEGeneric" ,
  eff_charge_cut_lowY = 0.0 ,
  Alpha2Order =  True 
)

from RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi import TTRHBuilderAngleAndTemplate
hltESPTTRHBuilderAngleAndTemplate = TTRHBuilderAngleAndTemplate.clone(
  StripCPE = "hltESPStripCPEfromTrackAngle" ,
  Matcher = "StandardMatcher" ,
  ComputeCoarseLocalPositionFromDisk = False ,
  PixelCPE = "hltESPPixelCPETemplateReco",
  ComponentName ="hltESPTTRHBuilderAngleAndTemplate" 
)
hltESPTTRHBWithTrackAngle = TTRHBuilderAngleAndTemplate.clone(
  StripCPE = "hltESPStripCPEfromTrackAngle",
  Matcher = "StandardMatcher",
  ComputeCoarseLocalPositionFromDisk = False,
  PixelCPE = "hltESPPixelCPEGeneric",
  ComponentName = "hltESPTTRHBWithTrackAngle" 
)

from RecoLocalTracker.SiStripRecHitConverter.StripCPEESProducer_cfi import stripCPEESProducer
hltESPStripCPEfromTrackAngle = stripCPEESProducer.clone(
  ComponentType = "StripCPEfromTrackAngle" ,
  ComponentName = "hltESPStripCPEfromTrackAngle",
  parameters = cms.PSet( 
    mLC_P2 = cms.double(0.3),
    mLC_P1 = cms.double(0.618),
    mLC_P0 = cms.double(-0.326),
#    useLegacyError = cms.bool( True ), # 50ns menu
#    maxChgOneMIP = cms.double( -6000.0 ), # 50ns menu
    useLegacyError = cms.bool(False) , # 25ns menu
    maxChgOneMIP = cms.double(6000.0) , #25ns menu
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

from RecoTracker.TkNavigation.NavigationSchoolESProducer_cfi import navigationSchoolESProducer
navigationSchoolESProducer = navigationSchoolESProducer.clone(
  ComponentName = "SimpleNavigationSchool" ,
  SimpleMagneticField ="ParabolicMf" 
)

from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import MeasurementTracker
hltESPMeasurementTracker = MeasurementTracker.clone(
  UseStripStripQualityDB =  True ,
  StripCPE = "hltESPStripCPEfromTrackAngle",
  UsePixelROCQualityDB = True ,
  DebugPixelROCQualityDB = False,
  UseStripAPVFiberQualityDB = True ,
  badStripCuts = cms.PSet(
    TOB = cms.PSet(
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet(
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet(
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet(
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = False ,
  ComponentName ="hltESPMeasurementTracker",
  DebugPixelModuleQualityDB = False,
  UsePixelModuleQualityDB = True,
  DebugStripAPVFiberQualityDB = False,
  HitMatcher = "StandardMatcher",
  DebugStripStripQualityDB = False,
  PixelCPE = "hltESPPixelCPEGeneric",
  SiStripQualityLabel = "" ,
  UseStripModuleQualityDB = True,
  MaskBadAPVFibers = True
)

hltESPRungeKuttaTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( True )
)

from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import KFUpdatorESProducer
hltESPKFUpdator = KFUpdatorESProducer.clone(
  ComponentName = "hltESPKFUpdator" 
)

hltESPChi2MeasurementEstimator30 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 10.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator30" ),
  nSigma = cms.double( 3.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MaxChi2 = cms.double( 30.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 )
)

from TrackingTools.TrackFitters.KFTrajectoryFitter_cfi import KFTrajectoryFitter
hltESPTrajectoryFitterRK = KFTrajectoryFitter.clone(
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectoryFitterRK" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)

hltSiStripExcludedFEDListProducer = cms.EDProducer( "SiStripExcludedFEDListProducer",
    ProductLabel = cms.InputTag( "rawDataCollector" )
)

hltMeasurementTrackerEvent = cms.EDProducer( "MeasurementTrackerEventProducer",
    inactivePixelDetectorLabels = cms.VInputTag(  ),
    stripClusterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
    pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
    switchOffPixelsIfEmpty = cms.bool( True ),
    inactiveStripDetectorLabels = cms.VInputTag( ),
    skipClusters = cms.InputTag( "" ),
    measurementTracker = cms.string( "hltESPMeasurementTracker" )
)

from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
pp_on_PbPb_run3.toModify(hltMeasurementTrackerEvent,
                         stripClusterProducer = cms.string( "hltHITrackingSiStripRawToClustersFacilityFullZeroSuppression" ),
                         pixelClusterProducer = cms.string( "hltSiPixelClustersAfterSplittingPPOnAA" ),
                         )

#####
hltESPTrajectoryFitterRK = cms.ESProducer( "KFTrajectoryFitterESProducer",
  appendToDataLabel = cms.string( "" ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectoryFitterRK" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
hltESPTrajectorySmootherRK = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectorySmootherRK" ),
  appendToDataLabel = cms.string( "" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
hltESPFittingSmootherIT = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  MinNumberOfHits = cms.int32( 3 ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  MaxFractionOutliers = cms.double( 0.3 ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  ComponentName = cms.string( "hltESPFittingSmootherIT" ),
  RejectTracks = cms.bool( True )
)

from DQMOffline.Trigger.SiStrip_OfflineMonitoring_cff import *
hltTrackRefitterForSiStripMonitorTrack.TTRHBuilder             = 'hltESPTTRHBWithTrackAngle'
hltTrackRefitterForSiStripMonitorTrack.Propagator              = 'hltESPRungeKuttaTrackerPropagator'
hltTrackRefitterForSiStripMonitorTrack.Fitter                  = 'hltESPFittingSmootherIT'
hltTrackRefitterForSiStripMonitorTrack.MeasurementTrackerEvent = 'hltMeasurementTrackerEvent'
hltTrackRefitterForSiStripMonitorTrack.NavigationSchool        = 'navigationSchoolESProducer'
hltTrackRefitterForSiStripMonitorTrack.src                     = 'hltMergedTracks' # hltIter2Merged

pp_on_PbPb_run3.toModify(hltTrackRefitterForSiStripMonitorTrack,
                         src = 'hltMergedTracksPPOnAA')

HLTSiStripMonitorTrack.TopFolderName = 'HLT/SiStrip'
HLTSiStripMonitorTrack.TrackProducer = 'hltTrackRefitterForSiStripMonitorTrack'
HLTSiStripMonitorTrack.TrackLabel    = ''
HLTSiStripMonitorTrack.Cluster_src   = 'hltSiStripRawToClustersFacility'
HLTSiStripMonitorTrack.AlgoName      = 'HLT'
HLTSiStripMonitorTrack.Trend_On      = True
HLTSiStripMonitorTrack.Mod_On        = False
HLTSiStripMonitorTrack.OffHisto_On   = True
HLTSiStripMonitorTrack.HistoFlag_On  = False
HLTSiStripMonitorTrack.TkHistoMap_On = False

pp_on_PbPb_run3.toModify(HLTSiStripMonitorTrack,
                         Cluster_src = "hltHITrackingSiStripRawToClustersFacilityFullZeroSuppression")

pp_on_PbPb_run3.toModify(HLTSiStripMonitorCluster,
                         BPTXfilter = dict(l1Algorithms = ['L1_ZeroBias']))

HLTSiStripMonitorClusterAPVgainCalibration = HLTSiStripMonitorCluster.clone()
from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *
#HLTSiStripMonitorClusterAPVgainCalibration.BPTXfilter = genericTriggerEventFlag4fullTrackerAndHLTnoHIPnoOOTdb # HLT_ZeroBias_FirstCollisionAfterAbortGap_*
HLTSiStripMonitorClusterAPVgainCalibration.BPTXfilter = cms.PSet(
   andOr         = cms.bool( False ),
### DCS selection
   dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
   dcsRecordInputTag   = cms.InputTag( "onlineMetaDataDigis" ),
   dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ),
   andOrDcs      = cms.bool( False ),
   errorReplyDcs = cms.bool( True ),
### HLT selection
   andOrHlt      = cms.bool(True),# True:=OR; False:=AND
   hltInputTag   = cms.InputTag( "TriggerResults::HLT" ),
   hltPaths      = cms.vstring("HLT_ZeroBias_FirstCollisionAfterAbortGap_v*"),
   errorReplyHlt = cms.bool( False ),
#   errorReplyHlt = cms.bool( True ),
### L1 selection
#   andOrL1       = cms.bool( True ),
#   l1Algorithms  = cms.vstring("L1_ZeroBias_FirstCollidingBunch"),
##   l1BeforeMask  = cms.bool( True ), # specifies, if the L1 algorithm decision should be read as before (true) or after (false) masking is applied.
#   l1BeforeMask  = cms.bool( False ), # specifies, if the L1 algorithm decision should be read as before (true) or after (false) masking is applied.
#   errorReplyL1  = cms.bool( False ),   
   verbosityLevel = cms.uint32(1)
)
HLTSiStripMonitorClusterAPVgainCalibration.TopFolderName = cms.string('HLT/SiStrip/ZeroBias_FirstCollisionAfterAbortGap')

pp_on_PbPb_run3.toModify(HLTSiStripMonitorClusterAPVgainCalibration,
                         BPTXfilter = dict(hltPaths = ["HLT_HICentrality30100_FirstCollisionAfterAbortGap_v*"]),
                         TopFolderName = cms.string('HLT/SiStrip/HLT_HICentrality30100_FirstCollisionAfterAbortGap'))

sistripOnlineMonitorHLTsequence = cms.Sequence(
    hltMeasurementTrackerEvent
    * sistripMonitorHLTsequence # strip cluster monitoring
    * HLTSiStripMonitorClusterAPVgainCalibration
)
