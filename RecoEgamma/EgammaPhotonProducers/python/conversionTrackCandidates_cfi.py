import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  $Id: conversionTrackCandidates.cfi,v 1.23 2008/05/16 18:00:50 nancy Exp $
#
# Tracker geometry #####################
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
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
conversionTrackCandidates = cms.EDProducer("ConversionTrackCandidateProducer",
    #    string  bcProducer  =        "islandBasicClusters"
    #    string  bcBarrelCollection = "islandBarrelBasicClusters"
    #    string  bcEndcapCollection = "islandEndcapBasicClusters"
    scHybridBarrelProducer = cms.string('correctedHybridSuperClusters'),
    inOutTrackCandidateSCAssociationCollection = cms.string('inOutTrackCandidateSCAssociationCollection'),
    maxHOverE = cms.double(0.2),
    scHybridBarrelCollection = cms.string(''),
    hbheModule = cms.string('hbhereco'),
    inOutTrackCandidateCollection = cms.string('inOutTracksFromConversions'),
    # old endcap clustering
    #    string scIslandEndcapProducer   =     "correctedEndcapSuperClustersWithPreshower"
    #    string scIslandEndcapCollection =     ""
    outInTrackCandidateCollection = cms.string('outInTracksFromConversions'),
    minSCEt = cms.double(5.0),
    MeasurementTrackerName = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('alongMomElePropagator'),
        propagatorOppositeTISE = cms.string('oppositeToMomElePropagator')
    ),
    InOutRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    bcEndcapCollection = cms.string('multi5x5EndcapBasicClusters'),
    outInTrackCandidateSCAssociationCollection = cms.string('outInTrackCandidateSCAssociationCollection'),
    bcBarrelCollection = cms.string('multi5x5BarrelBasicClusters'),
    scIslandEndcapProducer = cms.string('multi5x5SuperClustersWithPreshower'),
    bcProducer = cms.string('multi5x5BasicClusters'),
    OutInRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    scIslandEndcapCollection = cms.string(''),
    hbheInstance = cms.string(''),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForConversions'),
    hOverEConeSize = cms.double(0.1)
)


