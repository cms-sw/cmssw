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
CTF_TIFTOB_MeasurementTracker = copy.deepcopy(MeasurementTracker)
# trajectory filtering
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
ckfBaseTrajectoryFilterTIFTOB = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#replace ckfBaseTrajectoryFilterTIFTOB.filterPset.maxLostHits = 3
#replace ckfBaseTrajectoryFilterTIFTOB.filterPset.maxConsecLostHits = 1
#replace ckfBaseTrajectoryFilterTIFTOB.filterPset.minimumNumberOfHits =  4
#
GroupedCkfTrajectoryBuilderTIFTOB = copy.deepcopy(GroupedCkfTrajectoryBuilder)
CTF_TIFTOB_MeasurementTracker.ComponentName = 'CTF_TIFTOB'
CTF_TIFTOB_MeasurementTracker.pixelClusterProducer = ''
ckfBaseTrajectoryFilterTIFTOB.ComponentName = 'ckfBaseTrajectoryFilterTIFTOB'
ckfBaseTrajectoryFilterTIFTOB.filterPset.minPt = 0.01
GroupedCkfTrajectoryBuilderTIFTOB.MeasurementTrackerName = 'CTF_TIFTOB'
GroupedCkfTrajectoryBuilderTIFTOB.ComponentName = 'GroupedCkfTrajectoryBuilderTIFTOB'
GroupedCkfTrajectoryBuilderTIFTOB.trajectoryFilterName = 'ckfBaseTrajectoryFilterTIFTOB'

