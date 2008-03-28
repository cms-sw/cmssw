import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
#include "Geometry/CMSCommonData/data/cmsMagneticFieldXML.cfi"
from MagneticField.Engine.uniformMagneticField_cfi import *
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
CTF_TIFTIB_MeasurementTracker = copy.deepcopy(MeasurementTracker)
# trajectory filtering
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
ckfBaseTrajectoryFilterTIFTIB = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#
GroupedCkfTrajectoryBuilderTIFTIB = copy.deepcopy(GroupedCkfTrajectoryBuilder)
CTF_TIFTIB_MeasurementTracker.ComponentName = 'CTF_TIFTIB'
CTF_TIFTIB_MeasurementTracker.pixelClusterProducer = ''
ckfBaseTrajectoryFilterTIFTIB.ComponentName = 'ckfBaseTrajectoryFilterTIFTIB'
ckfBaseTrajectoryFilterTIFTIB.filterPset.minPt = 0.01
#replace ckfBaseTrajectoryFilterTIFTIB.filterPset.maxLostHits = 3
#replace ckfBaseTrajectoryFilterTIFTIB.filterPset.maxConsecLostHits = 1
ckfBaseTrajectoryFilterTIFTIB.filterPset.minimumNumberOfHits = 4
GroupedCkfTrajectoryBuilderTIFTIB.MeasurementTrackerName = 'CTF_TIFTIB'
GroupedCkfTrajectoryBuilderTIFTIB.ComponentName = 'GroupedCkfTrajectoryBuilderTIFTIB'
GroupedCkfTrajectoryBuilderTIFTIB.trajectoryFilterName = 'ckfBaseTrajectoryFilterTIFTIB'

