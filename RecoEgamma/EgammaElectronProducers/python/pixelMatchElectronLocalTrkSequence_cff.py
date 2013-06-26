import FWCore.ParameterSet.Config as cms

#
# create a sequence with all required modules and sources needed to make
# pixel based electrons
#
# $Id: pixelMatchElectronLocalTrkSequence_cff.py,v 1.6 2009/02/09 17:30:06 chamont Exp $
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
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# create sequence for tracking
pixelMatchElectronLocalTrkSequence = cms.Sequence(siStripClusters*siStripMatchedRecHits)

