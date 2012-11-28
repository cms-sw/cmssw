import FWCore.ParameterSet.Config as cms

### STEP 0 ###

# seeding layers
#import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
#initialLayerList = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.pixellayertriplets.clone(
#    ComponentName = 'initialLayerList'
#    )

# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeInitialSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeInitialSeeds.firstHitSubDetectorNumber = [1]
iterativeInitialSeeds.firstHitSubDetectors = [1]
iterativeInitialSeeds.secondHitSubDetectorNumber = [2]
iterativeInitialSeeds.secondHitSubDetectors = [1, 2]
iterativeInitialSeeds.thirdHitSubDetectorNumber = [2]
iterativeInitialSeeds.thirdHitSubDetectors = [1, 2]
iterativeInitialSeeds.seedingAlgo = ['InitialPixelTriplets']
iterativeInitialSeeds.minRecHits = [3] 
iterativeInitialSeeds.pTMin = [0.3]
iterativeInitialSeeds.maxD0 = [1.]
iterativeInitialSeeds.maxZ0 = [30.]
iterativeInitialSeeds.numberOfHits = [3]
iterativeInitialSeeds.originRadius = [1.0] # note: standard tracking uses 0.03, but this value gives a much better agreement in rate and shape for iter0
iterativeInitialSeeds.originHalfLength = [15.9] 
iterativeInitialSeeds.originpTMin = [0.6] 
iterativeInitialSeeds.zVertexConstraint = [-1.0]
iterativeInitialSeeds.primaryVertices = ['none']
# new (AG)
iterativeInitialSeeds.newSyntax = False
iterativeInitialSeeds.layerList = ['BPix1+BPix2+BPix3',
                                   'BPix1+BPix2+FPix1_pos',
                                   'BPix1+BPix2+FPix1_neg',
                                   'BPix1+FPix1_pos+FPix2_pos',
                                   'BPix1+FPix1_neg+FPix2_neg']

# candidate producer
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeInitialTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeInitialTrackCandidates.SeedProducer = cms.InputTag("iterativeInitialSeeds","InitialPixelTriplets")
iterativeInitialTrackCandidates.TrackProducers = ['globalPixelWithMaterialTracks']
iterativeInitialTrackCandidates.MinNumberOfCrossedLayers = 3

# track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeInitialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeInitialTracks.src = 'iterativeInitialTrackCandidates'
iterativeInitialTracks.TTRHBuilder = 'WithoutRefit'
iterativeInitialTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativeInitialTracks.Propagator = 'PropagatorWithMaterial'

# track merger
initialStepTracks = cms.EDProducer("FastTrackMerger",
                                   TrackProducers = cms.VInputTag(cms.InputTag("iterativeInitialTrackCandidates"),
                                                                  cms.InputTag("iterativeInitialTracks")),
                                   trackAlgo = cms.untracked.uint32(4) # iter0
                                   )

# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
initialStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='initialStepTracks',
        trackSelectors= cms.VPSet(
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
                name = 'initialStepLoose',
                            ), #end of pset
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
                name = 'initialStepTight',
                            preFilterName = 'initialStepLoose',
                            ),
                    RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
                name = 'initialStep',
                            preFilterName = 'initialStepTight',
                            ),
            ) #end of vpset
        ) #end of clone


# Final sequence
iterativeInitialStep = cms.Sequence(iterativeInitialSeeds
                                    +iterativeInitialTrackCandidates
                                    +iterativeInitialTracks
                                    +initialStepTracks
                                    +initialStepSelector)


