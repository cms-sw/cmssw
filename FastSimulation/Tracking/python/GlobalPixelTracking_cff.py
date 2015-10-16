import FWCore.ParameterSet.Config as cms

###################
# pixel tracks
# TOTO: where is this used, what fullsim products does it represent, is the cfg proper
###################

## seeds
from FastSimulation.Tracking.GlobalPixelSeedProducer_cff import globalPixelSeeds

## track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
globalPixelTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("globalPixelSeeds")
    )

## tracks
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
globalPixelWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone(
    src = 'globalPixelTrackCandidates',
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherWithOutlierRejection',
    Propagator = 'PropagatorWithMaterial',
    TrajectoryInEvent = cms.bool(True),
    )

###################
# pixel track candidates for electrons
# TOTO: where is this used, what fullsim products does it represent, is the cfg proper
###################

# masks
# introduction of this mask is based on review of GlobalPixelTracking_cff in CMSSW_7_2_2
# it doesn't necessarily make sense...
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
globalPixelMasksForElectrons = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.fastTrackerRecHitMaskProducer.clone(
    trajectories = cms.InputTag("globalPixelWithMaterialTracks")
    )

# seeds
from FastSimulation.Tracking.GlobalPixelSeedProducerForElectrons_cff import globalPixelSeedsForElectrons
globalPixelSeedsForElectrons.hitMasks = cms.InputTag("globalPixelMasksForElectrons")

# track candidates
# TODO: need masks?
globalPixelTrackCandidatesForElectrons = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("globalPixelSeedsForElectrons"),
    hitMasks = cms.InputTag("globalPixelMasksForElectrons")
    )

# tracks    
globalPixelWithMaterialTracksForElectrons = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone(
    src = 'globalPixelTrackCandidatesForElectrons',
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherWithOutlierRejection',
    Propagator = 'PropagatorWithMaterial',
    TrajectoryInEvent = cms.bool(True)
    )

###################
# pixel track candidates for photons
###################

# seed
from FastSimulation.Tracking.GlobalPixelSeedProducerForElectrons_cff import globalPixelSeedsForPhotons
globalPixelSeedsForPhotons.hitMasks = cms.InputTag("globalPixelMasksForElectrons")

# track candidate
globalPixelTrackCandidatesForPhotons = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("globalPixelSeedsForPhotons"),
    hitMasks = cms.InputTag("globalPixelMasksForElectrons")
    )

# tracks
globalPixelWithMaterialTracksForPhotons = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone(
    src = 'globalPixelTrackCandidatesForPhotons',
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherWithOutlierRejection',
    Propagator = 'PropagatorWithMaterial',
    TrajectoryInEvent = cms.bool(True)
)

####################

# The sequences
globalPixelTracking = cms.Sequence(globalPixelSeeds*
                                   globalPixelTrackCandidates*
                                   globalPixelWithMaterialTracks*
                                   globalPixelMasksForElectrons*
                                   globalPixelSeedsForElectrons*
                                   globalPixelTrackCandidatesForElectrons*
                                   globalPixelWithMaterialTracksForElectrons*
                                   globalPixelSeedsForPhotons*
                                   globalPixelTrackCandidatesForPhotons*
                                   globalPixelWithMaterialTracksForPhotons)
