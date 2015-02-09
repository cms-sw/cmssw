import FWCore.ParameterSet.Config as cms

# step 1

# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativePixelPairSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativePixelPairSeeds.simTrackSelection.skipSimTrackIdTags = [cms.InputTag("initialStepIds"), cms.InputTag("lowPtTripletStepIds")]
iterativePixelPairSeeds.outputSeedCollectionName = 'PixelPair'
iterativePixelPairSeeds.minRecHits =3
iterativePixelPairSeeds.simTrackSelection.pTMin = 0.3
iterativePixelPairSeeds.simTrackSelection.maxD0 = 5.
iterativePixelPairSeeds.simTrackSelection.maxZ0 = 50.
iterativePixelPairSeeds.numberOfHits = 2
iterativePixelPairSeeds.originRadius = 0.2
iterativePixelPairSeeds.originHalfLength = 17.5
iterativePixelPairSeeds.originpTMin = 0.6
iterativePixelPairSeeds.zVertexConstraint = -1.0
iterativePixelPairSeeds.primaryVertex = 'pixelVertices' # this is currently the only iteration why uses a PV instead of the BeamSpot 

#iterativePixelPairSeeds.layerList = ['BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 
#                                     'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 
#                                     'BPix1+FPix2_pos', 'BPix1+FPix2_neg', 
#                                     'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
#                                     'BPix2+FPix2_pos', 'BPix2+FPix2_neg', 
#                                     'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg', 
#                                     'FPix2_pos+TEC1_pos', 'FPix2_pos+TEC2_pos', 
#                                     'FPix2_neg+TEC1_neg', 'FPix2_neg+TEC2_neg']
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepSeedLayers
iterativePixelPairSeeds.layerList = pixelPairStepSeedLayers.layerList

# candidate producer
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativePixelPairCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativePixelPairCandidates.SeedProducer = cms.InputTag("iterativePixelPairSeeds","PixelPair")
iterativePixelPairCandidates.MinNumberOfCrossedLayers = 2 # ?

# track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativePixelPairTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativePixelPairTracks.src = 'iterativePixelPairCandidates'
iterativePixelPairTracks.TTRHBuilder = 'WithoutRefit'
iterativePixelPairTracks.Fitter = 'KFFittingSmootherSecond'
iterativePixelPairTracks.Propagator = 'PropagatorWithMaterial'
iterativePixelPairTracks.AlgorithmName = cms.string('pixelPairStep')

# simtrack id producer
pixelPairStepIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("iterativePixelPairTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )

# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
pixelPairStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='iterativePixelPairTracks',
            trackSelectors= cms.VPSet(
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
                name = 'pixelPairStepLoose',
                            ), #end of pset
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
                name = 'pixelPairStepTight',
                            preFilterName = 'pixelPairStepLoose',
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
                name = 'pixelPairStep',
                            preFilterName = 'pixelPairStepTight',
                            ),
                    ) #end of vpset
            ) #end of clone

# sequence
iterativePixelPairStep = cms.Sequence(iterativePixelPairSeeds+
                                      iterativePixelPairCandidates+
                                      iterativePixelPairTracks+
                                      pixelPairStepIds+
                                      pixelPairStepSelector)
