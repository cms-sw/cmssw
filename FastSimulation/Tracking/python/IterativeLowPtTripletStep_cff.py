import FWCore.ParameterSet.Config as cms

# step 0.5

# seeding
#from FastSimulation.Tracking.IterativeSecondSeedProducer_cff import *
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
iterativeLowPtTripletSeeds.pTMin = [0.1] 
iterativeLowPtTripletSeeds.maxD0 = [5.]
iterativeLowPtTripletSeeds.maxZ0 = [50.]
iterativeLowPtTripletSeeds.numberOfHits = [3]
iterativeLowPtTripletSeeds.originRadius = [0.03]
iterativeLowPtTripletSeeds.originHalfLength = [17.5] # ?
iterativeLowPtTripletSeeds.originpTMin = [0.2]
iterativeLowPtTripletSeeds.zVertexConstraint = [-1.0]
iterativeLowPtTripletSeeds.primaryVertices = ['none']

# candidate producer
#from FastSimulation.Tracking.IterativeLowPtTripletCandidateProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeLowPtTripletTrackCandidatesWithTriplets = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeLowPtTripletTrackCandidates = cms.Sequence(iterativeLowPtTripletTrackCandidatesWithTriplets)
iterativeLowPtTripletTrackCandidatesWithTriplets.SeedProducer = cms.InputTag("iterativeLowPtTripletSeeds","LowPtPixelTriplets")
iterativeLowPtTripletTrackCandidatesWithTriplets.TrackProducers = ['zeroStepFilter']
iterativeLowPtTripletTrackCandidatesWithTriplets.KeepFittedTracks = False
iterativeLowPtTripletTrackCandidatesWithTriplets.MinNumberOfCrossedLayers = 3

# track producer
#from FastSimulation.Tracking.IterativeLowPtTripletTrackProducer_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeLowPtTripletTracksWithTriplets = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeLowPtTripletTracks = cms.Sequence(iterativeLowPtTripletTracksWithTriplets)
iterativeLowPtTripletTracksWithTriplets.src = 'iterativeLowPtTripletTrackCandidatesWithTriplets'
iterativeLowPtTripletTracksWithTriplets.TTRHBuilder = 'WithoutRefit'
#iterativeLowPtTripletTracksWithTriplets.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativeLowPtTripletTracksWithTriplets.Fitter = 'KFFittingSmootherSecond'
iterativeLowPtTripletTracksWithTriplets.Propagator = 'PropagatorWithMaterial'

# track merger
#from FastSimulation.Tracking.IterativeLowPtTripletTrackMerger_cfi import *
iterativeLowPtTripletTrackMerging = cms.EDProducer("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativeLowPtTripletTrackCandidatesWithTriplets"),
                                   cms.InputTag("iterativeLowPtTripletTracksWithTriplets")),
    RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("zeroStepFilter")),
    trackAlgo = cms.untracked.uint32(5), # iter1
    MinNumberOfTrajHits = cms.untracked.uint32(3),
    MaxLostTrajHits = cms.untracked.uint32(1)
)

# track filter
#from FastSimulation.Tracking.IterativeLowPtTripletTrackFilter_cff import *
##import RecoParticleFlow.PFTracking.vertexFilter_cfi
##secStep = RecoParticleFlow.PFTracking.vertexFilter_cfi.vertFilter.clone()
##iterativeLowPtTripletTrackFiltering = cms.Sequence(secStep)
##secStep.recTracks = cms.InputTag("iterativeLowPtTripletTrackMerging")
##secStep.TrackAlgorithm = 'iter2'

# track selection
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

zerofiveStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeLowPtTripletTrackMerging',
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

zerofiveStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeLowPtTripletTrackMerging',
copyTrajectories = True,
copyExtras = True,
chi2n_par = 0.5,
res_par = ( 0.003, 0.001 ),
minNumberLayers = 5,
minNumber3DLayers = 3,
maxNumberLostLayers = 1,
d0_par1 = ( 0.9, 4.0 ),
dz_par1 = ( 0.9, 4.0 ),
d0_par2 = ( 0.9, 4.0 ),
dz_par2 = ( 0.9, 4.0 )
)

##import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi
##zerofiveStep = RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi.ctfrsTrackListMerger.clone()
##zerofiveStep.TrackProducer1 = 'zerofiveStepVtx'
##zerofiveStep.TrackProducer2 = 'zerofiveStepTrk'

zerofiveStep = cms.EDProducer("FastTrackMerger",
                       TrackProducers = cms.VInputTag(cms.InputTag("zerofiveStepVtx"),
                                                     cms.InputTag("zerofiveStepTrk"))
)

zerofivefilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("zerofiveStep")
)

iterativeLowPtTripletTrackFiltering = cms.Sequence(zerofiveStepVtx*zerofiveStepTrk*zerofiveStep*zerofivefilter)


# sequence
iterativeLowPtTripletStep = cms.Sequence(iterativeLowPtTripletSeeds+iterativeLowPtTripletTrackCandidatesWithTriplets+iterativeLowPtTripletTracks+iterativeLowPtTripletTrackMerging+iterativeLowPtTripletTrackFiltering)


