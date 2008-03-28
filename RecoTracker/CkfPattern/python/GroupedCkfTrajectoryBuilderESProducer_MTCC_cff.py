import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
#include "MagneticField/Engine/data/volumeBasedMagneticField.cfi"
from MagneticField.Engine.uniformMagneticField_cfi import *
# initialize geometry #####################
#include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi"
#include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi"
#include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"
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
# MeasurementTracker
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
#
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *

