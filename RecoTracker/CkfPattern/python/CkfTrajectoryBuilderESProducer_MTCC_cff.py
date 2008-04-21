import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
#include "MagneticField/Engine/data/volumeBasedMagneticField.cfi"
# initialize geometry #####################
#include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi"
#include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi"
#include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"
#include "RecoTracker/GeometryESProducer/data/TrackerRecoGeometryESProducer.cfi"
# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# pixelCPE
#include "RecoLocalTracker/SiPixelRecHits/data/PixelCPEParmError.cfi"
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# MeasurementTracker
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cff import *
#
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *

