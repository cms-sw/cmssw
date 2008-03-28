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
CTF_TIFTIBTOB_MeasurementTracker = copy.deepcopy(MeasurementTracker)
# trajectory filtering
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
ckfBaseTrajectoryFilterTIFTIBTOB = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#replace ckfBaseTrajectoryFilterTIFTIBTOB.filterPset.maxLostHits = 3
#replace ckfBaseTrajectoryFilterTIFTIBTOB.filterPset.maxConsecLostHits = 1
#replace ckfBaseTrajectoryFilterTIFTIBTOB.filterPset.minimumNumberOfHits =  4
#
GroupedCkfTrajectoryBuilderTIFTIBTOB = copy.deepcopy(GroupedCkfTrajectoryBuilder)
CTF_TIFTIBTOB_MeasurementTracker.ComponentName = 'CTF_TIFTIBTOB'
CTF_TIFTIBTOB_MeasurementTracker.pixelClusterProducer = ''
ckfBaseTrajectoryFilterTIFTIBTOB.ComponentName = 'ckfBaseTrajectoryFilterTIFTIBTOB'
ckfBaseTrajectoryFilterTIFTIBTOB.filterPset.minPt = 0.01
GroupedCkfTrajectoryBuilderTIFTIBTOB.MeasurementTrackerName = 'CTF_TIFTIBTOB'
GroupedCkfTrajectoryBuilderTIFTIBTOB.ComponentName = 'GroupedCkfTrajectoryBuilderTIFTIBTOB'
GroupedCkfTrajectoryBuilderTIFTIBTOB.trajectoryFilterName = 'ckfBaseTrajectoryFilterTIFTIBTOB'

