import FWCore.ParameterSet.Config as cms

# step 0.5

# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeLowPtTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeLowPtTripletSeeds.firstHitSubDetectorNumber = [1]
iterativeLowPtTripletSeeds.firstHitSubDetectors = [1]
iterativeLowPtTripletSeeds.secondHitSubDetectorNumber = [2]
iterativeLowPtTripletSeeds.secondHitSubDetectors = [1, 2]
iterativeLowPtTripletSeeds.thirdHitSubDetectorNumber = [2]
iterativeLowPtTripletSeeds.thirdHitSubDetectors = [1, 2]
iterativeLowPtTripletSeeds.seedingAlgo = ['LowPtPixelTriplets']
iterativeLowPtTripletSeeds.minRecHits = [3]
iterativeLowPtTripletSeeds.pTMin = [0.25] 
iterativeLowPtTripletSeeds.maxD0 = [5.]
iterativeLowPtTripletSeeds.maxZ0 = [50.]
iterativeLowPtTripletSeeds.numberOfHits = [3]
iterativeLowPtTripletSeeds.originRadius = [0.03]
iterativeLowPtTripletSeeds.originHalfLength = [17.5] # ?
iterativeLowPtTripletSeeds.originpTMin = [0.35]
iterativeLowPtTripletSeeds.zVertexConstraint = [-1.0]
iterativeLowPtTripletSeeds.primaryVertices = ['none']

iterativeLowPtTripletSeeds.newSyntax = True
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
iterativeLowPtTripletTrackCandidatesWithTriplets.TrackProducers = ['initialStep']
iterativeLowPtTripletTrackCandidatesWithTriplets.KeepFittedTracks = False
iterativeLowPtTripletTrackCandidatesWithTriplets.MinNumberOfCrossedLayers = 3

# track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeLowPtTripletTracksWithTriplets = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeLowPtTripletTracks = cms.Sequence(iterativeLowPtTripletTracksWithTriplets)
iterativeLowPtTripletTracksWithTriplets.src = 'iterativeLowPtTripletTrackCandidatesWithTriplets'
iterativeLowPtTripletTracksWithTriplets.TTRHBuilder = 'WithoutRefit'
iterativeLowPtTripletTracksWithTriplets.Fitter = 'KFFittingSmootherSecond'
iterativeLowPtTripletTracksWithTriplets.Propagator = 'PropagatorWithMaterial'

# track merger
lowPtTripletStepTracks = cms.EDProducer("FastTrackMerger",
                                        TrackProducers = cms.VInputTag(cms.InputTag("iterativeLowPtTripletTrackCandidatesWithTriplets"),
                                                                       cms.InputTag("iterativeLowPtTripletTracksWithTriplets")),
                                        RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("initialStep")),
                                        trackAlgo = cms.untracked.uint32(5), # iter1
                                        MinNumberOfTrajHits = cms.untracked.uint32(3),
                                        MaxLostTrajHits = cms.untracked.uint32(1)
                                        )

# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='lowPtTripletStepTracks',
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
                                         lowPtTripletStepTracks+
                                         lowPtTripletStepSelector)

