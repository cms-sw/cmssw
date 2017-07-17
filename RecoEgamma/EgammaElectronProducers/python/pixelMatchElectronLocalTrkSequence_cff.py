import FWCore.ParameterSet.Config as cms

#
# create a sequence with all required modules and sources needed to make
# pixel based electrons
#
#

# SiliconStrip Clusterizer and RecHit producer modules
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_SimData_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
# StripCPEfromTrackAngleESProducer es_module
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
# TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# CKFTrackCandidateMaker
# broken down to the different components so to configure the Chi2MeasurementEstimatorESProducer
#include "RecoTracker/CkfPattern/data/CkfTrackCandidates.cff"
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.TrackFitters.KFTrajectoryFitter_cfi import *
from TrackingTools.TrackFitters.KFTrajectorySmoother_cfi import *
from TrackingTools.TrackFitters.KFFittingSmoother_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# create sequence for tracking
pixelMatchElectronLocalTrkSequence = cms.Sequence(siStripClusters*siStripMatchedRecHits)

