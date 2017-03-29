import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################

# initialize geometry #####################

# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
Chi2MeasurementEstimatorForP5 = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone()
Chi2MeasurementEstimatorForP5.ComponentName = 'Chi2MeasurementEstimatorForP5'
Chi2MeasurementEstimatorForP5.MaxChi2 = 100.
Chi2MeasurementEstimatorForP5.nSigma = 4.
Chi2MeasurementEstimatorForP5.MaxDisplacement = 100
Chi2MeasurementEstimatorForP5.MaxSagitta=-1

# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi

## MeasurementTracker
##CTF_P5_MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
#replace CTF_P5_MeasurementTracker.pixelClusterProducer = ""

# trajectory filtering
from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import *
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
ckfBaseTrajectoryFilterP5 = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
ckfBaseTrajectoryFilterP5.minPt = 0.5
ckfBaseTrajectoryFilterP5.maxLostHits = 4
ckfBaseTrajectoryFilterP5.maxConsecLostHits = 3
#replace ckfBaseTrajectoryFilterP5.minimumNumberOfHits =  4
#
GroupedCkfTrajectoryBuilderP5 = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone()
##CTF_P5_MeasurementTracker.ComponentName = 'CTF_P5' # useless duplication of MeasurementTracker
##GroupedCkfTrajectoryBuilderP5.MeasurementTrackerName = 'CTF_P5' # useless duplication of MeasurementTracker
GroupedCkfTrajectoryBuilderP5.trajectoryFilter.refToPSet_ = 'ckfBaseTrajectoryFilterP5'
GroupedCkfTrajectoryBuilderP5.maxCand = 1
GroupedCkfTrajectoryBuilderP5.estimator = cms.string('Chi2MeasurementEstimatorForP5')
