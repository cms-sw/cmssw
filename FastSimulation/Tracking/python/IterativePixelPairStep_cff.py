import FWCore.ParameterSet.Config as cms

# trajectory seeds

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativePixelPairSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativePixelPairSeeds.simTrackSelection.skipSimTrackIds = [
        cms.InputTag("initialStepIds"), 
        cms.InputTag("detachedTripletStepIds"), 
        cms.InputTag("lowPtTripletStepIds")]
]
iterativePixelPairSeeds.simTrackSelection.pTMin = 0.3
iterativePixelPairSeeds.simTrackSelection.maxD0 = 5.
iterativePixelPairSeeds.simTrackSelection.maxZ0 = 50.
iterativePixelPairSeeds.minLayersCrossed =3
iterativePixelPairSeeds.originRadius = 0.2
iterativePixelPairSeeds.originHalfLength = 17.5
iterativePixelPairSeeds.originpTMin = 0.6

iterativePixelPairSeeds.beamSpot = ''
iterativePixelPairSeeds.primaryVertex = 'firstStepPrimaryVertices' # vertices are generated from the initalStepTracks
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepSeedLayers
iterativePixelPairSeeds.layerList = pixelPairStepSeedLayers.layerList

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativePixelPairCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativePixelPairCandidates.SeedProducer = cms.InputTag("iterativePixelPairSeeds")
iterativePixelPairCandidates.MinNumberOfCrossedLayers = 2 # ?

# tracks

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativePixelPairTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativePixelPairTracks.src = 'iterativePixelPairCandidates'
iterativePixelPairTracks.TTRHBuilder = 'WithoutRefit'
iterativePixelPairTracks.Fitter = 'KFFittingSmootherSecond'
iterativePixelPairTracks.Propagator = 'PropagatorWithMaterial'
iterativePixelPairTracks.AlgorithmName = cms.string('pixelPairStep')

# track identification

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
pixelPairStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='iterativePixelPairTracks',
            trackSelectors= cms.VPSet(
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
                name = 'pixelPairStepLoose',
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
                name = 'pixelPairStepTight',
                            preFilterName = 'pixelPairStepLoose',
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
                name = 'pixelPairStep',
                            preFilterName = 'pixelPairStepTight',
                            ),
                    )
            )

# simtrack id producer

pixelPairStepIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("iterativePixelPairTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )

# sequence

iterativePixelPairStep = cms.Sequence(iterativePixelPairSeeds+
                                      iterativePixelPairCandidates+
                                      iterativePixelPairTracks+
                                      pixelPairStepIds+
                                      pixelPairStepSelector)
