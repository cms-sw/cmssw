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
CTF_TIFTOBTEC_MeasurementTracker = copy.deepcopy(MeasurementTracker)
# trajectory filtering
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
ckfBaseTrajectoryFilterTIFTOBTEC = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#replace ckfBaseTrajectoryFilterTIFTOBTEC.filterPset.maxLostHits = 3
#replace ckfBaseTrajectoryFilterTIFTOBTEC.filterPset.maxConsecLostHits = 1
#replace ckfBaseTrajectoryFilterTIFTOBTEC.filterPset.minimumNumberOfHits =  4
#
GroupedCkfTrajectoryBuilderTIFTOBTEC = copy.deepcopy(GroupedCkfTrajectoryBuilder)
CTF_TIFTOBTEC_MeasurementTracker.ComponentName = 'CTF_TIFTOBTEC'
CTF_TIFTOBTEC_MeasurementTracker.pixelClusterProducer = ''
ckfBaseTrajectoryFilterTIFTOBTEC.ComponentName = 'ckfBaseTrajectoryFilterTIFTOBTEC'
ckfBaseTrajectoryFilterTIFTOBTEC.filterPset.minPt = 0.01
GroupedCkfTrajectoryBuilderTIFTOBTEC.MeasurementTrackerName = 'CTF_TIFTOBTEC'
GroupedCkfTrajectoryBuilderTIFTOBTEC.ComponentName = 'GroupedCkfTrajectoryBuilderTIFTOBTEC'
GroupedCkfTrajectoryBuilderTIFTOBTEC.trajectoryFilterName = 'ckfBaseTrajectoryFilterTIFTOBTEC'

