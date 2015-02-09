import FWCore.ParameterSet.Config as cms

# step 0.5

# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeLowPtTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeLowPtTripletSeeds.skipSimTrackIdTags = [
    cms.InputTag("initialStepIds"),
    cms.InputTag("detachedTripletStepIds")]
iterativeLowPtTripletSeeds.outputSeedCollectionName = 'LowPtPixelTriplets'
iterativeLowPtTripletSeeds.minRecHits = 3
iterativeLowPtTripletSeeds.pTMin = 0.25
iterativeLowPtTripletSeeds.maxD0 = 5.
iterativeLowPtTripletSeeds.maxZ0 = 50.
iterativeLowPtTripletSeeds.numberOfHits = 3
iterativeLowPtTripletSeeds.originRadius = 0.03
iterativeLowPtTripletSeeds.originHalfLength = 17.5
iterativeLowPtTripletSeeds.originpTMin = 0.35
iterativeLowPtTripletSeeds.zVertexConstraint = -1.0
iterativeLowPtTripletSeeds.primaryVertex = 'none'

#iterativeLowPtTripletSeeds.layerList = ['BPix1+BPix2+BPix3',
#                                   'BPix1+BPix2+FPix1_pos',
#                                   'BPix1+BPix2+FPix1_neg',
#                                   'BPix1+FPix1_pos+FPix2_pos',
#                                   'BPix1+FPix1_neg+FPix2_neg']
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
iterativeLowPtTripletSeeds.layerList = PixelLayerTriplets.layerList

# candidate producer

import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeLowPtTripletTrackCandidatesWithTriplets = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeLowPtTripletTrackCandidates = cms.Sequence(iterativeLowPtTripletTrackCandidatesWithTriplets)
iterativeLowPtTripletTrackCandidatesWithTriplets.SeedProducer = cms.InputTag("iterativeLowPtTripletSeeds","LowPtPixelTriplets")
iterativeLowPtTripletTrackCandidatesWithTriplets.MinNumberOfCrossedLayers = 3

# track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeLowPtTripletTracksWithTriplets = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeLowPtTripletTracks = cms.Sequence(iterativeLowPtTripletTracksWithTriplets)
iterativeLowPtTripletTracksWithTriplets.src = 'iterativeLowPtTripletTrackCandidatesWithTriplets'
iterativeLowPtTripletTracksWithTriplets.TTRHBuilder = 'WithoutRefit'
iterativeLowPtTripletTracksWithTriplets.Fitter = 'KFFittingSmootherSecond'
iterativeLowPtTripletTracksWithTriplets.Propagator = 'PropagatorWithMaterial'
iterativeLowPtTripletTracksWithTriplets.AlgorithmName = cms.string('lowPtTripletStep')


# simtrack id producer
lowPtTripletStepIds = cms.EDProducer("SimTrackIdProducer",
                                     trackCollection = cms.InputTag("iterativeLowPtTripletTracksWithTriplets"),
                                     HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                     )

# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='iterativeLowPtTripletTracksWithTriplets',
            trackSelectors= cms.VPSet(
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
                name = 'lowPtTripletStepLoose',
                            ), #end of pset
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
                name = 'lowPtTripletStepTight',
                            preFilterName = 'lowPtTripletStepLoose',
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
                name = 'lowPtTripletStep',
                            preFilterName = 'lowPtTripletStepTight',
                            ),
                    ) #end of vpset
            ) #end of clone


iterativeLowPtTripletStep = cms.Sequence(iterativeLowPtTripletSeeds+
                                         iterativeLowPtTripletTrackCandidatesWithTriplets+
                                         iterativeLowPtTripletTracks+  
                                         lowPtTripletStepIds+
                                         lowPtTripletStepSelector)

