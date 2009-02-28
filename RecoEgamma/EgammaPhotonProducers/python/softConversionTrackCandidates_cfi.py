import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  $Id: softConversionTrackCandidates_cfi.py,v 1.3 2008/07/14 14:01:31 nancy Exp $
#
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
    #clusterType = cms.string('BasicCluster'),
    clusterType = cms.string('pfCluster'),
    MeasurementTrackerName = cms.string(''),
#    clusterBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    clusterBarrelCollection = cms.InputTag("particleFlowClusterECAL"),
    clusterEndcapCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    outInTrackCandidateClusterAssociationCollection = cms.string('outInTrackCandidateClusterAssociationCollection'),
    InOutRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    OutInRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    outInTrackCandidateCollection = cms.string('softOITrackCandidates'),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForConversions')
)



