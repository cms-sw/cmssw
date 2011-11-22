import FWCore.ParameterSet.Config as cms

# step 4

# seeding
#from FastSimulation.Tracking.IterativeFourthSeedProducer_cff import *
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativePixelLessSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativePixelLessSeeds.firstHitSubDetectorNumber = [3]
iterativePixelLessSeeds.firstHitSubDetectors = [3, 4, 6]
iterativePixelLessSeeds.secondHitSubDetectorNumber = [3]
iterativePixelLessSeeds.secondHitSubDetectors = [3, 4, 6]
iterativePixelLessSeeds.thirdHitSubDetectorNumber = [0]
iterativePixelLessSeeds.thirdHitSubDetectors = []
iterativePixelLessSeeds.seedingAlgo = ['PixelLessPairs']
###iterativePixelLessSeeds.minRecHits = [5]
iterativePixelLessSeeds.minRecHits = [3]
iterativePixelLessSeeds.pTMin = [0.3]
#cut on fastsim simtracks. I think it should be removed for the 4th step
#iterativePixelLessSeeds.maxD0 = [20.]
#iterativePixelLessSeeds.maxZ0 = [50.]
iterativePixelLessSeeds.maxD0 = [99.]
iterativePixelLessSeeds.maxZ0 = [99.]
#-----
iterativePixelLessSeeds.numberOfHits = [2]
#values for the seed compatibility constraint
iterativePixelLessSeeds.originRadius = [2.0]
iterativePixelLessSeeds.originHalfLength = [10.0]
iterativePixelLessSeeds.originpTMin = [0.6] # was 0.5
iterativePixelLessSeeds.zVertexConstraint = [-1.0]
iterativePixelLessSeeds.primaryVertices = ['none']

# candidate producer
#from FastSimulation.Tracking.IterativeFourthCandidateProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativePixelLessTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativePixelLessTrackCandidates.SeedProducer = cms.InputTag("iterativePixelLessSeeds","PixelLessPairs")
iterativePixelLessTrackCandidates.TrackProducers = ['firstfilter', 'secfilter','thfilter'] # add 0 and 0.5 ?
iterativePixelLessTrackCandidates.KeepFittedTracks = False
iterativePixelLessTrackCandidates.MinNumberOfCrossedLayers = 6 # was 5


# track producer
#from FastSimulation.Tracking.IterativeFourthTrackProducer_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativePixelLessTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativePixelLessTracks.src = 'iterativePixelLessTrackCandidates'
iterativePixelLessTracks.TTRHBuilder = 'WithoutRefit'
##iterativePixelLessTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativePixelLessTracks.Fitter = 'KFFittingSmootherFourth'
iterativePixelLessTracks.Propagator = 'PropagatorWithMaterial'


# track merger
#from FastSimulation.Tracking.IterativeFourthTrackMerger_cfi import *
iterativePixelLessTrackMerging = cms.EDProducer("FastTrackMerger",
TrackProducers = cms.VInputTag(cms.InputTag("iterativePixelLessTrackCandidates"),
                               cms.InputTag("iterativePixelLessTracks")),
RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("zeroStepFilter"),
                                                cms.InputTag("zerofivefilter"),   
                                                cms.InputTag("firstfilter"),   
                                                cms.InputTag("secfilter"),     
                                                cms.InputTag("thfilter")),     
trackAlgo = cms.untracked.uint32(9),
MinNumberOfTrajHits = cms.untracked.uint32(6), # was 5
MaxLostTrajHits = cms.untracked.uint32(0)
)


# track filter
#from FastSimulation.Tracking.IterativeFourthTrackFilter_cff import *
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
fouStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativePixelLessTrackMerging',
##keepAllTracks = True
copyExtras = True,
copyTrajectories = True,
chi2n_par = 0.3,
res_par = ( 0.003, 0.001 ),
minNumberLayers = 6,
minNumber3DLayers = 3,
maxNumberLostLayers = 0,
d0_par1 = ( 1.0, 4.0 ),
dz_par1 = ( 1.0, 4.0 ),
d0_par2 = ( 1.0, 4.0 ),
dz_par2 = ( 1.0, 4.0 )
)

foufilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("fouStep")
)


iterativePixelLessTrackFiltering = cms.Sequence(fouStep*foufilter)


# sequence
iterativePixelLessStep = cms.Sequence(iterativePixelLessSeeds+iterativePixelLessTrackCandidates+iterativePixelLessTracks+iterativePixelLessTrackMerging+iterativePixelLessTrackFiltering)

