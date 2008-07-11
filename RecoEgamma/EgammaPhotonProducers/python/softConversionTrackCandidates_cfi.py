import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  $Id: softConversionTrackCandidates_cfi.py,v 1.1 2008/05/28 03:45:36 dwjang Exp $
#
# Tracker geometry #####################
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# MeasurementTracker
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
# TransientTracks
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
#TrajectoryFilter
from RecoEgamma.EgammaPhotonProducers.trajectoryFilterForConversions_cfi import *
#TrajectoryBuilder
from RecoEgamma.EgammaPhotonProducers.trajectoryBuilderForConversions_cfi import *
softConversionTrackCandidates = cms.EDProducer("SoftConversionTrackCandidateProducer",
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('alongMomElePropagator'),
        propagatorOppositeTISE = cms.string('oppositeToMomElePropagator')
    ),
    inOutTrackCandidateCollection = cms.string('softIOTrackCandidates'),
    inOutTrackCandidateClusterAssociationCollection = cms.string('inOutTrackCandidateClusterAssociationCollection'),
    clusterBarrelCollection = cms.string('islandBarrelBasicClusters'),
    clusterType = cms.string('BasicCluster'),
    MeasurementTrackerName = cms.string(''),
    clusterProducer = cms.string('islandBasicClusters'),
    clusterEndcapCollection = cms.string('islandEndcapBasicClusters'),
    outInTrackCandidateClusterAssociationCollection = cms.string('outInTrackCandidateClusterAssociationCollection'),
    InOutRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    OutInRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    outInTrackCandidateCollection = cms.string('softOITrackCandidates'),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForConversions')
)



