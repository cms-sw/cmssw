import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
#  $Id: conversionTrackCandidates_cfi.py,v 1.28 2011/03/04 12:40:58 sani Exp $
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
#TrajectoryCleaning
from RecoEgamma.EgammaPhotonProducers.trajectoryCleanerBySharedHitsForConversions_cfi import *
#Propagators
from RecoEgamma.EgammaPhotonProducers.propAlongMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.propOppoMomentumWithMaterialForElectrons_cfi import *


conversionTrackCandidates = cms.EDProducer("ConversionTrackCandidateProducer",
#    beamSpot = cms.InputTag("offlineBeamSpot"),
    bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    bcEndcapCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters"),
    scIslandEndcapProducer = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"),
    outInTrackCandidateSCAssociationCollection = cms.string('outInTrackCandidateSCAssociationCollection'),                                           
    inOutTrackCandidateSCAssociationCollection = cms.string('inOutTrackCandidateSCAssociationCollection'),
    inOutTrackCandidateCollection = cms.string('inOutTracksFromConversions'),
    outInTrackCandidateCollection = cms.string('outInTracksFromConversions'),
    MeasurementTrackerName = cms.string(''),
    OutInRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    InOutRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    useHitsSplitting = cms.bool(False),
    maxNumOfSeedsOutIn = cms.int32(50),
    maxNumOfSeedsInOut = cms.int32(50),                                       
    hcalTowers = cms.InputTag("towerMaker"),                                       
    minSCEt = cms.double(10.0),
    hOverEConeSize = cms.double(0.15),
    maxHOverE = cms.double(0.5),
    fractionShared = cms.double(0.5),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForConversions'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('alongMomElePropagator'),
        propagatorOppositeTISE = cms.string('oppositeToMomElePropagator'),
        numberMeasurementsForFit = cms.int32(4)
    ),
    allowSharedFirstHit = cms.bool(False),
    ValidHitBonus = cms.double(999.0),
    MissingHitPenalty = cms.double(0.0)
)


