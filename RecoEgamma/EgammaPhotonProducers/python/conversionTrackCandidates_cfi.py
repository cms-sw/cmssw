import FWCore.ParameterSet.Config as cms

#
#  configuration for producer of converted photons
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

from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import *
from RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi import *

conversionTrackCandidates = cms.EDProducer("ConversionTrackCandidateProducer",
#    beamSpot = cms.InputTag("offlineBeamSpot"),
    bcBarrelCollection = cms.InputTag('particleFlowSuperClusterECAL:particleFlowBasicClusterECALBarrel'),
    bcEndcapCollection = cms.InputTag('particleFlowSuperClusterECAL:particleFlowBasicClusterECALEndcap'),
    scHybridBarrelProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel'),
    scIslandEndcapProducer = cms.InputTag('particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower'),
    outInTrackCandidateSCAssociationCollection = cms.string('outInTrackCandidateSCAssociationCollection'),                                           
    inOutTrackCandidateSCAssociationCollection = cms.string('inOutTrackCandidateSCAssociationCollection'),
    inOutTrackCandidateCollection = cms.string('inOutTracksFromConversions'),
    outInTrackCandidateCollection = cms.string('outInTracksFromConversions'),
    barrelEcalRecHitCollection = cms.InputTag('ecalRecHit:EcalRecHitsEB'),
    endcapEcalRecHitCollection = cms.InputTag('ecalRecHit:EcalRecHitsEE'),
    MeasurementTrackerName = cms.string(''),
    OutInRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    InOutRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    useHitsSplitting = cms.bool(False),
    maxNumOfSeedsOutIn = cms.int32(50),
    maxNumOfSeedsInOut = cms.int32(50),
    bcEtCut = cms.double(1.5),
    bcECut  = cms.double(1.5),
    useEtCut = cms.bool(True),                                         
    hcalTowers = cms.InputTag("towerMaker"),                                       
    minSCEt = cms.double(20.0),
    hOverEConeSize = cms.double(0.15),
    maxHOverE = cms.double(0.15),
    isoInnerConeR =  cms.double(3.5),
    isoConeR =  cms.double(0.4),
    isoEtaSlice =  cms.double(2.5),
    isoEtMin = cms.double(0.0),
    isoEMin = cms.double(0.08),
    vetoClusteredHits  = cms.bool(False),
    useNumXstals = cms.bool(True),
    ecalIsoCut_offset =  cms.double(999999999),
    ecalIsoCut_slope  =  cms.double(0.),                                                   
#    ecalIsoCut_offset =  cms.double(4.2),
#    ecalIsoCut_slope =  cms.double(0.003),                                                   
    
    RecHitFlagToBeExcludedEB = cleanedHybridSuperClusters.RecHitFlagToBeExcluded,
    RecHitSeverityToBeExcludedEB = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    RecHitFlagToBeExcludedEE = multi5x5BasicClustersCleaned.RecHitFlagToBeExcluded,
    RecHitSeverityToBeExcludedEE = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
                                                                               
    fractionShared = cms.double(0.5),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForConversions'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('alongMomElePropagator'),
        propagatorOppositeTISE = cms.string('oppositeToMomElePropagator'),
        numberMeasurementsForFit = cms.int32(4)
    ),
    allowSharedFirstHit = cms.bool(True),
    ValidHitBonus = cms.double(5.0),
    MissingHitPenalty = cms.double(20.0)

 )


