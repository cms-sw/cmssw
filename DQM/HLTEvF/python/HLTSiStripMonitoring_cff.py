import FWCore.ParameterSet.Config as cms

from TrackingTools.RecoGeometry.RecoGeometries_cff import *
hltESPDummyDetLayerGeometry = DummyDetLayerGeometry.clone(
    ComponentName = cms.string( "hltESPDummyDetLayerGeometry" )
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
  DoCosmics = cms.bool( False ),
  LoadTemplatesFromDB = cms.bool( True ),
  ComponentName = cms.string( "hltESPPixelCPETemplateReco" ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  speed = cms.int32( -2 ),
  UseClusterSplitter = cms.bool( False )
)

from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import PixelCPEGenericESProducer
hltESPPixelCPEGeneric = PixelCPEGenericESProducer.clone(
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
  TanLorentzAnglePerTesla = cms.double( 0.106 ),
  inflate_errors = cms.bool( False ),
  eff_charge_cut_lowX = cms.double( 0.0 ),
  eff_charge_cut_highY = cms.double( 1.0 ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  EdgeClusterErrorY = cms.double( 85.0 ),
  ComponentName = cms.string( "hltESPPixelCPEGeneric" ),
  eff_charge_cut_lowY = cms.double( 0.0 ),
  PixelErrorParametrization = cms.string( "NOTcmsim" ),
  Alpha2Order = cms.bool( True )
)

from RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi import TTRHBuilderAngleAndTemplate
hltESPTTRHBuilderAngleAndTemplate = TTRHBuilderAngleAndTemplate.clone(
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPETemplateReco" ),
  ComponentName = cms.string( "hltESPTTRHBuilderAngleAndTemplate" )
)
hltESPTTRHBWithTrackAngle = TTRHBuilderAngleAndTemplate.clone(
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBWithTrackAngle" )
)

from RecoLocalTracker.SiStripRecHitConverter.StripCPEESProducer_cfi import stripCPEESProducer
hltESPStripCPEfromTrackAngle = stripCPEESProducer.clone(
  ComponentType = cms.string( "StripCPEfromTrackAngle" ),
  ComponentName = cms.string( "hltESPStripCPEfromTrackAngle" ),
  parameters = cms.PSet( 
    mLC_P2 = cms.double( 0.3 ),
    mLC_P1 = cms.double( 0.618 ),
    mLC_P0 = cms.double( -0.326 ),
#    useLegacyError = cms.bool( True ), # 50ns menu
#    maxChgOneMIP = cms.double( -6000.0 ), # 50ns menu
    useLegacyError = cms.bool( False ), # 25ns menu
    maxChgOneMIP = cms.double( 6000.0 ), #25ns menu
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
  ComponentName = cms.string( "SimpleNavigationSchool" ),
  SimpleMagneticField = cms.string( "ParabolicMf" )
)

from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import MeasurementTracker
hltESPMeasurementTracker = MeasurementTracker.clone(
  UseStripStripQualityDB = cms.bool( True ),
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
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
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  UsePixelModuleQualityDB = cms.bool( True ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  SiStripQualityLabel = cms.string( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  MaskBadAPVFibers = cms.bool( True )
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
  ComponentName = cms.string( "hltESPKFUpdator" )
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
hltTrackRefitterForSiStripMonitorTrack.TTRHBuilder             = cms.string('hltESPTTRHBWithTrackAngle')
hltTrackRefitterForSiStripMonitorTrack.Propagator              = cms.string('hltESPRungeKuttaTrackerPropagator')
hltTrackRefitterForSiStripMonitorTrack.Fitter                  = cms.string('hltESPFittingSmootherIT')
hltTrackRefitterForSiStripMonitorTrack.MeasurementTrackerEvent = cms.InputTag('hltMeasurementTrackerEvent')
hltTrackRefitterForSiStripMonitorTrack.NavigationSchool        = cms.string('navigationSchoolESProducer')
hltTrackRefitterForSiStripMonitorTrack.src                     = cms.InputTag("hltTracksMerged") # hltIter2Merged

HLTSiStripMonitorTrack.TopFolderName = cms.string('HLT/SiStrip')
HLTSiStripMonitorTrack.TrackProducer = 'hltTrackRefitterForSiStripMonitorTrack'
HLTSiStripMonitorTrack.TrackLabel    = ''
HLTSiStripMonitorTrack.Cluster_src   = cms.InputTag('hltSiStripRawToClustersFacility')
HLTSiStripMonitorTrack.AlgoName      = cms.string("HLT")
HLTSiStripMonitorTrack.Trend_On      = cms.bool(True)
HLTSiStripMonitorTrack.Mod_On        = cms.bool(False)
HLTSiStripMonitorTrack.OffHisto_On   = cms.bool(True)
HLTSiStripMonitorTrack.HistoFlag_On  = cms.bool(False)
HLTSiStripMonitorTrack.TkHistoMap_On = cms.bool(False)

HLTSiStripMonitorClusterAPVgainCalibration = HLTSiStripMonitorCluster.clone()
from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *
#HLTSiStripMonitorClusterAPVgainCalibration.BPTXfilter = genericTriggerEventFlag4fullTrackerAndHLTnoHIPnoOOTdb # HLT_ZeroBias_FirstCollisionAfterAbortGap_*
HLTSiStripMonitorClusterAPVgainCalibration.BPTXfilter = cms.PSet(
   andOr         = cms.bool( False ),
### DCS selection
   dcsInputTag   = cms.InputTag( "scalersRawToDigi" ),
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

sistripOnlineMonitorHLTsequence = cms.Sequence(
    hltMeasurementTrackerEvent
    * sistripMonitorHLTsequence # strip cluster monitoring
    * HLTSiStripMonitorClusterAPVgainCalibration
)
