

import FWCore.ParameterSet.Config as cms

### ITERATIVE TRACKING: STEP 0 ###


# seeding
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeInitialSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeInitialSeeds.simTrackSelection.pTMin = 0.4 # it was 0.3
iterativeInitialSeeds.simTrackSelection.maxD0 = 1.
iterativeInitialSeeds.simTrackSelection.maxZ0 = 30.
iterativeInitialSeeds.minLayersCrossed = 3
iterativeInitialSeeds.outputSeedCollectionName = 'InitialPixelTriplets'
iterativeInitialSeeds.originRadius = 1.0 # note: standard tracking uses 0.03, but this value gives a much better agreement in rate and shape for iter0
iterativeInitialSeeds.originHalfLength = 999 # it was 15.9 
iterativeInitialSeeds.originpTMin = 0.6

iterativeInitialSeeds.primaryVertex = ''

#iterativeInitialSeeds.layerList = ['BPix1+BPix2+BPix3',
#                                   'BPix1+BPix2+FPix1_pos',
#                                   'BPix1+BPix2+FPix1_neg',
#                                   'BPix1+FPix1_pos+FPix2_pos',
#                                   'BPix1+FPix1_neg+FPix2_neg']
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
iterativeInitialSeeds.layerList = PixelLayerTriplets.layerList

# candidate producer
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeInitialTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeInitialTrackCandidates.SeedProducer = cms.InputTag("iterativeInitialSeeds",'InitialPixelTriplets')
iterativeInitialTrackCandidates.MinNumberOfCrossedLayers = 3

# track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeInitialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeInitialTracks.src = 'iterativeInitialTrackCandidates'
iterativeInitialTracks.TTRHBuilder = 'WithoutRefit'
iterativeInitialTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativeInitialTracks.Propagator = 'PropagatorWithMaterial'
iterativeInitialTracks.AlgorithmName = cms.string('initialStep')

#vertices
import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
firstStepPrimaryVertices=RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()
firstStepPrimaryVertices.TrackLabel = cms.InputTag("iterativeInitialTracks")
firstStepPrimaryVertices.vertexCollections = cms.VPSet(
    [cms.PSet(label=cms.string(""),
              algorithm=cms.string("AdaptiveVertexFitter"),
              minNdof=cms.double(0.0),
              useBeamConstraint = cms.bool(False),
              maxDistanceToBeam = cms.double(1.0)
              )
     ]
)

# simtrack id producer
initialStepIds = cms.EDProducer("SimTrackIdProducer",
                                trackCollection = cms.InputTag("iterativeInitialTracks"),
                                HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                )

# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
initialStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='iterativeInitialTracks',
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
                                    +firstStepPrimaryVertices
                                    +initialStepSelector
                                    +initialStepIds)




