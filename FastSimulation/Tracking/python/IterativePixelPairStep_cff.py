import FWCore.ParameterSet.Config as cms

# step 1

# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativePixelPairSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativePixelPairSeeds.firstHitSubDetectorNumber = [2]
iterativePixelPairSeeds.firstHitSubDetectors = [1, 2]
iterativePixelPairSeeds.secondHitSubDetectorNumber = [2]
iterativePixelPairSeeds.secondHitSubDetectors = [1, 2]
iterativePixelPairSeeds.thirdHitSubDetectorNumber = [2]
iterativePixelPairSeeds.thirdHitSubDetectors = [1, 2]
iterativePixelPairSeeds.seedingAlgo = ['PixelPair']
iterativePixelPairSeeds.minRecHits = [3]
iterativePixelPairSeeds.pTMin = [0.3]
iterativePixelPairSeeds.maxD0 = [5.]
iterativePixelPairSeeds.maxZ0 = [50.]
iterativePixelPairSeeds.numberOfHits = [2]
iterativePixelPairSeeds.originRadius = [0.2]
iterativePixelPairSeeds.originHalfLength = [17.5] 
iterativePixelPairSeeds.originpTMin = [0.6]
iterativePixelPairSeeds.zVertexConstraint = [-1.0]
iterativePixelPairSeeds.primaryVertices = ['pixelVertices']

# candidate producer
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativePixelPairCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativePixelPairCandidates.SeedProducer = cms.InputTag("iterativePixelPairSeeds","PixelPair")
iterativePixelPairCandidates.TrackProducers = ['lowPtTripletStepTracks']
iterativePixelPairCandidates.KeepFittedTracks = False
iterativePixelPairCandidates.MinNumberOfCrossedLayers = 2 # ?

# track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativePixelPairTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativePixelPairTracks.src = 'iterativePixelPairCandidates'
iterativePixelPairTracks.TTRHBuilder = 'WithoutRefit'
iterativePixelPairTracks.Fitter = 'KFFittingSmootherSecond'
iterativePixelPairTracks.Propagator = 'PropagatorWithMaterial'

# track merger
pixelPairStepTracks = cms.EDProducer("FastTrackMerger",
                                     TrackProducers = cms.VInputTag(cms.InputTag("iterativePixelPairCandidates"),
                                                                    cms.InputTag("iterativePixelPairTracks")),
                                     RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("initialStepTracks"),
                                                                                     cms.InputTag("lowPtTripletStepTracks")),
                                     trackAlgo = cms.untracked.uint32(6), # iter2
                                     MinNumberOfTrajHits = cms.untracked.uint32(3),
                                     MaxLostTrajHits = cms.untracked.uint32(1)
                                     )

# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
pixelPairStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='pixelPairStepTracks',
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
                                      pixelPairStepTracks+
                                      pixelPairStepSelector)


