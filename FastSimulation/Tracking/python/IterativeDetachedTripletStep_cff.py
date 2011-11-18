import FWCore.ParameterSet.Config as cms

# step 2

# seeding
#from FastSimulation.Tracking.IterativeSecondSeedProducer_cff import *
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeDetachedTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeDetachedTripletSeeds.firstHitSubDetectorNumber = [1]
iterativeDetachedTripletSeeds.firstHitSubDetectors = [1]
iterativeDetachedTripletSeeds.secondHitSubDetectorNumber = [2]
iterativeDetachedTripletSeeds.secondHitSubDetectors = [1, 2]
iterativeDetachedTripletSeeds.thirdHitSubDetectorNumber = [2]
iterativeDetachedTripletSeeds.thirdHitSubDetectors = [1, 2]
iterativeDetachedTripletSeeds.seedingAlgo = ['DetachedPixelTriplets']
iterativeDetachedTripletSeeds.minRecHits = [3]
iterativeDetachedTripletSeeds.pTMin = [0.1]
iterativeDetachedTripletSeeds.maxD0 = [5.]
iterativeDetachedTripletSeeds.maxZ0 = [50.]
iterativeDetachedTripletSeeds.numberOfHits = [3]
iterativeDetachedTripletSeeds.originRadius = [1.0] # was 0.2 cm
iterativeDetachedTripletSeeds.originHalfLength = [17.5] # ?
iterativeDetachedTripletSeeds.originpTMin = [0.2] # was 0.075 GeV
iterativeDetachedTripletSeeds.zVertexConstraint = [-1.0]
iterativeDetachedTripletSeeds.primaryVertices = ['none']

# candidate producer
#from FastSimulation.Tracking.IterativeSecondCandidateProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeDetachedTripletTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeDetachedTripletTrackCandidates.SeedProducer = cms.InputTag("iterativeDetachedTripletSeeds","DetachedPixelTriplets")
iterativeDetachedTripletTrackCandidates.TrackProducers = ['firstfilter']
iterativeDetachedTripletTrackCandidates.KeepFittedTracks = False
iterativeDetachedTripletTrackCandidates.MinNumberOfCrossedLayers = 3

# track producer
#from FastSimulation.Tracking.IterativeSecondTrackProducer_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeDetachedTripletTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeDetachedTripletTracks.src = 'iterativeDetachedTripletTrackCandidates'
iterativeDetachedTripletTracks.TTRHBuilder = 'WithoutRefit'
iterativeDetachedTripletTracks.Fitter = 'KFFittingSmootherSecond'
iterativeDetachedTripletTracks.Propagator = 'PropagatorWithMaterial'

# track merger
#from FastSimulation.Tracking.IterativeSecondTrackMerger_cfi import *
iterativeDetachedTripletTrackMerging = cms.EDProducer("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativeDetachedTripletTrackCandidates"),
                                   cms.InputTag("iterativeDetachedTripletTracks")),
    RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("zeroStepFilter"), 
                                   cms.InputTag("zerofivefilter"),
                                   cms.InputTag("firstfilter")),
    trackAlgo = cms.untracked.uint32(7), # iter3 
    MinNumberOfTrajHits = cms.untracked.uint32(3),
    MaxLostTrajHits = cms.untracked.uint32(1)
)


# track filter
#from FastSimulation.Tracking.IterativeSecondTrackFilter_cff import *
##import RecoParticleFlow.PFTracking.vertexFilter_cfi
##secStep = RecoParticleFlow.PFTracking.vertexFilter_cfi.vertFilter.clone()
##iterativeSecondTrackFiltering = cms.Sequence(secStep)
##secStep.recTracks = cms.InputTag("iterativeSecondTrackMerging")
##secStep.TrackAlgorithm = 'iter2'

# track selection
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

secStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeDetachedTripletTrackMerging',
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

secStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeDetachedTripletTrackMerging',
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
##secStep = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
##secStep.TrackProducer1 = 'secStepVtx'
##secStep.TrackProducer2 = 'secStepTrk'

secStep = cms.EDProducer("FastTrackMerger",
                       TrackProducers = cms.VInputTag(cms.InputTag("secStepVtx"),
                                                     cms.InputTag("secStepTrk"))
)

secfilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("secStep")
)

iterativeDetachedTripletTrackFiltering = cms.Sequence(secStepVtx*secStepTrk*secStep*secfilter)


# sequence
iterativeDetachedTripletStep = cms.Sequence(iterativeDetachedTripletSeeds+iterativeDetachedTripletTrackCandidates+iterativeDetachedTripletTracks+iterativeDetachedTripletTrackMerging+iterativeDetachedTripletTrackFiltering)


