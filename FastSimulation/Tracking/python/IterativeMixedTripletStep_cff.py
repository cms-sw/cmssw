import FWCore.ParameterSet.Config as cms

# step 3

# seeding
#from FastSimulation.Tracking.IterativeMixedTripletStepSeedProducer_cff import *
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeMixedTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeMixedTripletStepSeeds.firstHitSubDetectorNumber = [2]
##iterativeMixedTripletStepSeeds.firstHitSubDetectors = [1, 2, 6]
iterativeMixedTripletStepSeeds.firstHitSubDetectors = [1, 2]
iterativeMixedTripletStepSeeds.secondHitSubDetectorNumber = [3]
iterativeMixedTripletStepSeeds.secondHitSubDetectors = [1, 2, 6]
iterativeMixedTripletStepSeeds.thirdHitSubDetectorNumber = [0]
iterativeMixedTripletStepSeeds.thirdHitSubDetectors = []
iterativeMixedTripletStepSeeds.seedingAlgo = ['MixedTriplets']
iterativeMixedTripletStepSeeds.minRecHits = [3]
iterativeMixedTripletStepSeeds.pTMin = [0.15]
iterativeMixedTripletStepSeeds.maxD0 = [10.]
iterativeMixedTripletStepSeeds.maxZ0 = [30.]
iterativeMixedTripletStepSeeds.numberOfHits = [2]
iterativeMixedTripletStepSeeds.originRadius = [2.0] # was 1.2
iterativeMixedTripletStepSeeds.originHalfLength = [10.0] # was 7.0
iterativeMixedTripletStepSeeds.originpTMin = [0.35] # we need to add another seed for endcaps only, with 0.5
iterativeMixedTripletStepSeeds.zVertexConstraint = [-1.0]
iterativeMixedTripletStepSeeds.primaryVertices = ['none']


# candidate producer
#from FastSimulation.Tracking.IterativeThirdCandidateProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeMixedTripletStepCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeMixedTripletStepCandidates.SeedProducer = cms.InputTag("iterativeMixedTripletStepSeeds","MixedTriplets")
iterativeMixedTripletStepCandidates.TrackProducers = ['firstfilter', 'secfilter']
iterativeMixedTripletStepCandidates.KeepFittedTracks = False
iterativeMixedTripletStepCandidates.MinNumberOfCrossedLayers = 3


# track producer
#from FastSimulation.Tracking.IterativeThirdTrackProducer_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeMixedTripletStepTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeMixedTripletStepTracks.src = 'iterativeMixedTripletStepCandidates'
iterativeMixedTripletStepTracks.TTRHBuilder = 'WithoutRefit'
##iterativeMixedTripletStepTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativeMixedTripletStepTracks.Fitter = 'KFFittingSmootherThird'
iterativeMixedTripletStepTracks.Propagator = 'PropagatorWithMaterial'

# track merger
#from FastSimulation.Tracking.IterativeMixedTripletStepMerger_cfi import *
iterativeMixedTripletStepMerging = cms.EDProducer("FastTrackMerger",
                                          TrackProducers = cms.VInputTag(cms.InputTag("iterativeMixedTripletStepCandidates"),
                                                                         cms.InputTag("iterativeMixedTripletStepTracks")),
                                          RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("zeroStepFilter"),
                                                                                          cms.InputTag("zerofivefilter"),   
                                                                                          cms.InputTag("firstfilter"),   
                                                                                          cms.InputTag("secfilter")),    
                                          trackAlgo = cms.untracked.uint32(8),
                                          MinNumberOfTrajHits = cms.untracked.uint32(4), # ?
                                          MaxLostTrajHits = cms.untracked.uint32(0)                                          
                                          )

# track filter
#from FastSimulation.Tracking.IterativeMixedTripletStepFilter_cff import *
##OLD WAY
##import RecoParticleFlow.PFTracking.vertexFilter_cfi
##thStep = RecoParticleFlow.PFTracking.vertexFilter_cfi.vertFilter.clone()
##iterativeThirdTrackFiltering = cms.Sequence(thStep)
##thStep.recTracks = cms.InputTag("iterativeThirdTrackMerging")
##thStep.TrackAlgorithm = 'iter3'
##thStep.DistZFromVertex = 0.1

import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
thStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeMixedTripletStepMerging',
copyTrajectories = True,
copyExtras = True,
chi2n_par = 0.9,
res_par = ( 0.003, 0.001 ),
minNumberLayers = 3,
##minNumber3DLayers = 3,
minNumber3DLayers = 1, # ?
maxNumberLostLayers = 1,
d0_par1 = ( 0.9, 3.0 ),
dz_par1 = ( 0.9, 3.0 ),
d0_par2 = ( 1.0, 3.0 ),
dz_par2 = ( 1.0, 3.0 )
)

thStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeMixedTripletStepMerging',
copyTrajectories = True,
chi2n_par = 0.5,
res_par = ( 0.003, 0.001 ),
#minNumberLayers = 5,
minNumberLayers = 3,
#minNumber3DLayers = 4,
minNumber3DLayers = 1, # ?
maxNumberLostLayers = 1,
d0_par1 = ( 1.0, 4.0 ),
dz_par1 = ( 1.0, 4.0 ),
d0_par2 = ( 1.0, 4.0 ),
dz_par2 = ( 1.0, 4.0 )
)

##import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
##thStep = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
##thStep.TrackProducer1 = 'thStepVtx'
##thStep.TrackProducer2 = 'thStepTrk'


thStep = cms.EDProducer("FastTrackMerger",
                      TrackProducers = cms.VInputTag(cms.InputTag("thStepVtx"),
                                                     cms.InputTag("thStepTrk"))
)

thfilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("thStep")
)

iterativeMixedTripletStepFiltering = cms.Sequence(thStepVtx*thStepTrk*thStep*thfilter)

# sequence
iterativeMixedTripletStep = cms.Sequence(iterativeMixedTripletStepSeeds+iterativeMixedTripletStepCandidates+iterativeMixedTripletStepTracks+iterativeMixedTripletStepMerging+iterativeMixedTripletStepFiltering)

