import FWCore.ParameterSet.Config as cms

# trajectory seeds

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeLowPtTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeLowPtTripletSeeds.simTrackSelection.skipSimTrackIds = [    
    cms.InputTag("initialStepIds"),
    cms.InputTag("detachedTripletStepIds")]
]
iterativeLowPtTripletSeeds.simTrackSelection.pTMin = 0.25
iterativeLowPtTripletSeeds.simTrackSelection.maxD0 = 5.
iterativeLowPtTripletSeeds.simTrackSelection.maxZ0 = 50.
iterativeLowPtTripletSeeds.minLayersCrossed = 3
iterativeLowPtTripletSeeds.originRadius = 0.03
iterativeLowPtTripletSeeds.originHalfLength = 17.5
iterativeLowPtTripletSeeds.originpTMin = 0.35
iterativeLowPtTripletSeeds.primaryVertex = ''
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
iterativeLowPtTripletSeeds.layerList = PixelLayerTriplets.layerList

# track candidates

import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeLowPtTripletTrackCandidatesWithTriplets = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeLowPtTripletTrackCandidates = cms.Sequence(iterativeLowPtTripletTrackCandidatesWithTriplets)
iterativeLowPtTripletTrackCandidatesWithTriplets.SeedProducer = cms.InputTag("iterativeLowPtTripletSeeds")
iterativeLowPtTripletTrackCandidatesWithTriplets.MinNumberOfCrossedLayers = 3

# tracks

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeLowPtTripletTracksWithTriplets = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeLowPtTripletTracks = cms.Sequence(iterativeLowPtTripletTracksWithTriplets)
iterativeLowPtTripletTracksWithTriplets.src = 'iterativeLowPtTripletTrackCandidatesWithTriplets'
iterativeLowPtTripletTracksWithTriplets.TTRHBuilder = 'WithoutRefit'
iterativeLowPtTripletTracksWithTriplets.Fitter = 'KFFittingSmootherSecond'
iterativeLowPtTripletTracksWithTriplets.Propagator = 'PropagatorWithMaterial'
iterativeLowPtTripletTracksWithTriplets.AlgorithmName = cms.string('lowPtTripletStep')


                                     )

# track identification

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


# simtrack id producer

lowPtTripletStepIds = cms.EDProducer("SimTrackIdProducer",
                                     trackCollection = cms.InputTag("iterativeLowPtTripletTracksWithTriplets"),
                                     HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")

# final sequence

iterativeLowPtTripletStep = cms.Sequence(iterativeLowPtTripletSeeds+
                                         iterativeLowPtTripletTrackCandidatesWithTriplets+
                                         iterativeLowPtTripletTracks+  
                                         lowPtTripletStepIds+
                                         lowPtTripletStepSelector)

