import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  $Id: conversionTrackCandidates.cfi,v 1.25 2008/05/29 15:29:33 arizzi Exp $
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
    InOutRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    bcEndcapCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    outInTrackCandidateSCAssociationCollection = cms.string('outInTrackCandidateSCAssociationCollection'),
    #     string  bcProducer  =   "multi5x5BasicClusters"
    #     string  bcBarrelCollection = "multi5x5BarrelBasicClusters"
    #     string  bcEndcapCollection = "multi5x5EndcapBasicClusters"
    bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    scIslandEndcapProducer = cms.string('correctedMulti5x5SuperClustersWithPreshower'),
    OutInRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    hOverEConeSize = cms.double(0.1),
    scIslandEndcapCollection = cms.string(''),
    hbheInstance = cms.string(''),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForConversions'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('alongMomElePropagator'),
        propagatorOppositeTISE = cms.string('oppositeToMomElePropagator')
    )
)


