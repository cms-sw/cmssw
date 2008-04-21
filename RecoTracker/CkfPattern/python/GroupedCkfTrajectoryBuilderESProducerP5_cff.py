import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
#include "Geometry/CMSCommonData/data/cmsMagneticFieldXML.cfi"
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# initialize geometry #####################
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
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
import copy
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
# MeasurementTracker
CTF_P5_MeasurementTracker = copy.deepcopy(MeasurementTracker)
# trajectory filtering
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
ckfBaseTrajectoryFilterP5 = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#replace ckfBaseTrajectoryFilterP5.filterPset.minPt = 0.01
#replace ckfBaseTrajectoryFilterP5.filterPset.maxLostHits = 4
#replace ckfBaseTrajectoryFilterP5.filterPset.maxConsecLostHits = 3
#replace ckfBaseTrajectoryFilterP5.filterPset.minimumNumberOfHits =  4
#
GroupedCkfTrajectoryBuilderP5 = copy.deepcopy(GroupedCkfTrajectoryBuilder)
CTF_P5_MeasurementTracker.ComponentName = 'CTF_P5'
CTF_P5_MeasurementTracker.pixelClusterProducer = ''
ckfBaseTrajectoryFilterP5.ComponentName = 'ckfBaseTrajectoryFilterP5'
GroupedCkfTrajectoryBuilderP5.MeasurementTrackerName = 'CTF_P5'
GroupedCkfTrajectoryBuilderP5.ComponentName = 'GroupedCkfTrajectoryBuilderP5'
GroupedCkfTrajectoryBuilderP5.trajectoryFilterName = 'ckfBaseTrajectoryFilterP5'

