import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################

# initialize geometry #####################

# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
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
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
ckfBaseTrajectoryFilterP5 = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
ckfBaseTrajectoryFilterP5.filterPset.minPt = 0.5
ckfBaseTrajectoryFilterP5.filterPset.maxLostHits = 4
ckfBaseTrajectoryFilterP5.filterPset.maxConsecLostHits = 3
#replace ckfBaseTrajectoryFilterP5.filterPset.minimumNumberOfHits =  4
#
GroupedCkfTrajectoryBuilderP5 = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
ckfBaseTrajectoryFilterP5.ComponentName = 'ckfBaseTrajectoryFilterP5'
##CTF_P5_MeasurementTracker.ComponentName = 'CTF_P5' # useless duplication of MeasurementTracker
##GroupedCkfTrajectoryBuilderP5.MeasurementTrackerName = 'CTF_P5' # useless duplication of MeasurementTracker
GroupedCkfTrajectoryBuilderP5.ComponentName = 'GroupedCkfTrajectoryBuilderP5'
GroupedCkfTrajectoryBuilderP5.trajectoryFilterName = 'ckfBaseTrajectoryFilterP5'

