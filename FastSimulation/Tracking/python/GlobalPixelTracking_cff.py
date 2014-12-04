import FWCore.ParameterSet.Config as cms

# Global Pixel seeding
from FastSimulation.Tracking.GlobalPixelSeedProducer_cff import *
from FastSimulation.Tracking.GlobalPixelSeedProducerForElectrons_cff import *

# TrackCandidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
# reco::Tracks (possibly with invalid hits)
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

###################
globalPixelTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
globalPixelTrackCandidates.SeedProducer = cms.InputTag("globalPixelSeeds","GlobalPixel")

globalPixelWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
globalPixelWithMaterialTracks.src = 'globalPixelTrackCandidates'
globalPixelWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
globalPixelWithMaterialTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
globalPixelWithMaterialTracks.Propagator = 'PropagatorWithMaterial'
globalPixelWithMaterialTracks.TrajectoryInEvent = cms.bool(True)

# simtrack id producer
globalPixelStepIds = cms.EDProducer("SimTrackIdProducer",
                                     trackCollection = cms.InputTag("globalPixelWithMaterialTracks"),
                                   )

###################


globalPixelTrackCandidatesForElectrons = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
globalPixelTrackCandidatesForElectrons.SeedProducer = cms.InputTag("globalPixelSeedsForElectrons","GlobalPixel")
#globalPixelTrackCandidatesForElectrons.TrackProducers = cms.vstring(['globalPixelWithMaterialTracks'])
    
globalPixelWithMaterialTracksForElectrons = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
globalPixelWithMaterialTracksForElectrons.src = 'globalPixelTrackCandidatesForElectrons'
globalPixelWithMaterialTracksForElectrons.TTRHBuilder = 'WithoutRefit'
globalPixelWithMaterialTracksForElectrons.Fitter = 'KFFittingSmootherWithOutlierRejection'
globalPixelWithMaterialTracksForElectrons.Propagator = 'PropagatorWithMaterial'
globalPixelWithMaterialTracksForElectrons.TrajectoryInEvent = cms.bool(True)

####################

globalPixelTrackCandidatesForPhotons = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
globalPixelTrackCandidatesForPhotons.SeedProducer = cms.InputTag("globalPixelSeedsForPhotons","GlobalPixel")
#globalPixelTrackCandidatesForPhotons.TrackProducers = cms.vstring(['globalPixelWithMaterialTracks'])

globalPixelWithMaterialTracksForPhotons = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
globalPixelWithMaterialTracksForPhotons.src = 'globalPixelTrackCandidatesForPhotons'
globalPixelWithMaterialTracksForPhotons.TTRHBuilder = 'WithoutRefit'
globalPixelWithMaterialTracksForPhotons.Fitter = 'KFFittingSmootherWithOutlierRejection'
globalPixelWithMaterialTracksForPhotons.Propagator = 'PropagatorWithMaterial'
globalPixelWithMaterialTracksForPhotons.TrajectoryInEvent = cms.bool(True)

####################

# The sequences
globalPixelTracking = cms.Sequence(globalPixelSeeds*
                                   globalPixelTrackCandidates*
                                   globalPixelWithMaterialTracks*
                                   globalPixelStepIds*
                                   globalPixelSeedsForPhotons*
                                   globalPixelTrackCandidatesForPhotons*
                                   globalPixelWithMaterialTracksForPhotons*
                                   globalPixelSeedsForElectrons*
                                   globalPixelTrackCandidatesForElectrons*
                                   globalPixelWithMaterialTracksForElectrons)
