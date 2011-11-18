import FWCore.ParameterSet.Config as cms

# step 5

# seeding
#from FastSimulation.Tracking.IterativeFifthSeedProducer_cff import *
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeTobTecSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeTobTecSeeds.firstHitSubDetectorNumber = [2]
iterativeTobTecSeeds.firstHitSubDetectors = [5, 6]
iterativeTobTecSeeds.secondHitSubDetectorNumber = [2]
iterativeTobTecSeeds.secondHitSubDetectors = [5, 6]
iterativeTobTecSeeds.thirdHitSubDetectorNumber = [0]
iterativeTobTecSeeds.thirdHitSubDetectors = []
iterativeTobTecSeeds.seedingAlgo = ['TobTecLayerPairs']
iterativeTobTecSeeds.minRecHits = [4]
iterativeTobTecSeeds.pTMin = [0.3]
#cut on fastsim simtracks. I think it should be removed for the 5th step
iterativeTobTecSeeds.maxD0 = [99.]
iterativeTobTecSeeds.maxZ0 = [99.]
#-----
iterativeTobTecSeeds.numberOfHits = [2]
#values for the seed compatibility constraint
iterativeTobTecSeeds.originRadius = [6.0] # was 5.0
iterativeTobTecSeeds.originHalfLength = [30.0] # was 10.0
iterativeTobTecSeeds.originpTMin = [0.6] # was 0.5
iterativeTobTecSeeds.zVertexConstraint = [-1.0]
iterativeTobTecSeeds.primaryVertices = ['none']

# candidate producer
#from FastSimulation.Tracking.IterativeFifthCandidateProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeTobTecTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeTobTecTrackCandidates.SeedProducer = cms.InputTag("iterativeTobTecSeeds","TobTecLayerPairs")
iterativeTobTecTrackCandidates.TrackProducers = ['firstfilter','secfilter','thfilter','foufilter'] # add 0 and 0.5?
iterativeTobTecTrackCandidates.KeepFittedTracks = False
iterativeTobTecTrackCandidates.MinNumberOfCrossedLayers = 3


# track producer
#from FastSimulation.Tracking.IterativeFifthTrackProducer_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeTobTecTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeTobTecTracks.src = 'iterativeTobTecTrackCandidates'
iterativeTobTecTracks.TTRHBuilder = 'WithoutRefit'
iterativeTobTecTracks.Fitter = 'KFFittingSmootherFifth'
iterativeTobTecTracks.Propagator = 'PropagatorWithMaterial'


# track merger
#from FastSimulation.Tracking.IterativeFifthTrackMerger_cfi import *
iterativeTobTecTrackMerging = cms.EDProducer("FastTrackMerger",
                                          TrackProducers = cms.VInputTag(cms.InputTag("iterativeTobTecTrackCandidates"),
                                                                         cms.InputTag("iterativeTobTecTracks")),
                                          RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("zeroStepFilter"),
                                                                                          cms.InputTag("zerofivefilter"),  
                                                                                          cms.InputTag("firstfilter"),  
                                                                                          cms.InputTag("secfilter"),    
                                                                                          cms.InputTag("thfilter"),     
                                                                                          cms.InputTag("foufilter")),   
                                          trackAlgo = cms.untracked.uint32(10), # iter6
                                          MinNumberOfTrajHits = cms.untracked.uint32(6), # was 4
                                          MaxLostTrajHits = cms.untracked.uint32(0)
                                          )


# track filter
#from FastSimulation.Tracking.IterativeFifthTrackFilter_cff import *
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
fifthStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
src = 'iterativeTobTecTrackMerging',
##keepAllTracks = True,
copyExtras = True,
copyTrajectories = True,
chi2n_par = 0.25,
res_par = ( 0.003, 0.001 ),
minNumberLayers = 6, # was 4
minNumber3DLayers = 2,
maxNumberLostLayers = 0,
d0_par1 = ( 1.2, 4.0 ),
dz_par1 = ( 1.1, 4.0 ),
d0_par2 = ( 1.2, 4.0 ),
dz_par2 = ( 1.1, 4.0 )
)

fifthfilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("fifthStep")
)


iterativeTobTecTrackFiltering = cms.Sequence(fifthStep*fifthfilter)


# sequence
iterativeTobTecStep = cms.Sequence(iterativeTobTecSeeds
                                      +iterativeTobTecTrackCandidates
                                      +iterativeTobTecTracks
                                      +iterativeTobTecTrackMerging
                                      +iterativeTobTecTrackFiltering)

