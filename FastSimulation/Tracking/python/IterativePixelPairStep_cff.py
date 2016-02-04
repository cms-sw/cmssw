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
iterativePixelPairSeeds.originRadius = [0.01] # was 0.2
iterativePixelPairSeeds.originHalfLength = [17.5] # ?
iterativePixelPairSeeds.originpTMin = [0.6]
iterativePixelPairSeeds.zVertexConstraint = [-1.0]
iterativePixelPairSeeds.primaryVertices = ['pixelVertices']

# candidate producer
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativePixelPairCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativePixelPairCandidates.SeedProducer = cms.InputTag("iterativePixelPairSeeds","PixelPair")
iterativePixelPairCandidates.TrackProducers = ['zerofivefilter']
iterativePixelPairCandidates.KeepFittedTracks = False
iterativePixelPairCandidates.MinNumberOfCrossedLayers = 2 # ?

# track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativePixelPairTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativePixelPairTracks.src = 'iterativePixelPairCandidates'
iterativePixelPairTracks.TTRHBuilder = 'WithoutRefit'
#iterativePixelPairTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativePixelPairTracks.Fitter = 'KFFittingSmootherSecond'
iterativePixelPairTracks.Propagator = 'PropagatorWithMaterial'

# track merger
iterativePixelPairTrackMerging = cms.EDProducer("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativePixelPairCandidates"),
                                   cms.InputTag("iterativePixelPairTracks")),
    RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("zeroStepFilter"),
                                   cms.InputTag("zerofivefilter")),
    trackAlgo = cms.untracked.uint32(6), # iter2
    MinNumberOfTrajHits = cms.untracked.uint32(3),
    MaxLostTrajHits = cms.untracked.uint32(1)
)

# track filter
#from FastSimulation.Tracking.IterativeLowPtTripletTrackFilter_cff import *
##import RecoParticleFlow.PFTracking.vertexFilter_cfi
##secStep = RecoParticleFlow.PFTracking.vertexFilter_cfi.vertFilter.clone()
##iterativeLowPtTripletTrackFiltering = cms.Sequence(secStep)
##secStep.recTracks = cms.InputTag("iterativePixelPairTrackMerging")
##secStep.TrackAlgorithm = 'iter2'

# track selection
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

firstStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativePixelPairTrackMerging',
copyTrajectories = True,
copyExtras = True,
chi2n_par = 0.9,
res_par = ( 0.003, 0.001 ),
minNumberLayers = 3,
minNumber3DLayers = 3,
maxNumberLostLayers = 1,
d0_par1 = ( 0.85, 3.0 ),
dz_par1 = ( 0.8, 3.0 ),
d0_par2 = ( 0.9, 3.0 ),
dz_par2 = ( 0.9, 3.0 )
)

firstStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativePixelPairTrackMerging',
copyTrajectories = True,
copyExtras = True,
chi2n_par = 0.5,
res_par = ( 0.003, 0.001 ),
minNumberLayers = 3, # was 5
minNumber3DLayers = 3,
maxNumberLostLayers = 1,
d0_par1 = ( 0.9, 4.0 ),
dz_par1 = ( 0.9, 4.0 ),
d0_par2 = ( 0.9, 4.0 ),
dz_par2 = ( 0.9, 4.0 )
)

##import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
##firstStep = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
##firstStep.TrackProducer1 = 'firstStepVtx'
##firstStep.TrackProducer2 = 'firstStepTrk'

firstStep = cms.EDProducer("FastTrackMerger",
                       TrackProducers = cms.VInputTag(cms.InputTag("firstStepVtx"),
                                                     cms.InputTag("firstStepTrk"))
)

firstfilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("firstStep")
)

iterativePixelPairTrackFiltering = cms.Sequence(firstStepVtx*firstStepTrk*firstStep*firstfilter)


# sequence
iterativePixelPairStep = cms.Sequence(iterativePixelPairSeeds+iterativePixelPairCandidates+iterativePixelPairTracks+iterativePixelPairTrackMerging+iterativePixelPairTrackFiltering)


