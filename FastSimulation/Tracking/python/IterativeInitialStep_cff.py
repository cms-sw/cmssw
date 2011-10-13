import FWCore.ParameterSet.Config as cms

### STEP 0 ###

# seeding
#from FastSimulation.Tracking.IterativeFirstSeedProducer_cff import *
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
iterativeInitialSeeds.originRadius = [0.03] # new; was 0.2 cm
iterativeInitialSeeds.originHalfLength = [15.9] # ?
iterativeInitialSeeds.originpTMin = [0.6] # new; was 0.8
iterativeInitialSeeds.zVertexConstraint = [-1.0]
iterativeInitialSeeds.primaryVertices = ['none']

# candidate producer
#from FastSimulation.Tracking.IterativeFirstCandidateProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeInitialTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeInitialTrackCandidates.SeedProducer = cms.InputTag("iterativeInitialSeeds","InitialPixelTriplets")
iterativeInitialTrackCandidates.TrackProducers = ['globalPixelWithMaterialTracks']
iterativeInitialTrackCandidates.MinNumberOfCrossedLayers = 3

# track producer
#from FastSimulation.Tracking.IterativeFirstTrackProducer_cff import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeInitialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeInitialTracks.src = 'iterativeInitialTrackCandidates'
iterativeInitialTracks.TTRHBuilder = 'WithoutRefit'
iterativeInitialTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativeInitialTracks.Propagator = 'PropagatorWithMaterial'

# track merger
#from FastSimulation.Tracking.IterativeFirstTrackMerger_cfi import *
iterativeZeroTrackMerging = cms.EDProducer("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativeInitialTrackCandidates"),
                                   cms.InputTag("iterativeInitialTracks")),
    trackAlgo = cms.untracked.uint32(4) # iter0
)

# track filter
#from FastSimulation.Tracking.IterativeFirstTrackFilter_cff import *
# Track filtering and quality.
#   input:    iterativeFirstTrackMerging 
#   output:   generalTracks
#   sequence: iterativeFirstTrackFiltering
# Official sequence has loose and tight quality tracks, not reproduced
# here. (People will use generalTracks, eventually.)
###from RecoTracker.IterativeTracking.FirstFilter_cfi import *


import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi


zeroStepTracksWithQuality = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
zeroStepTracksWithQuality.src = 'iterativeZeroTrackMerging'
zeroStepTracksWithQuality.keepAllTracks = True
zeroStepTracksWithQuality.copyExtras = True
zeroStepTracksWithQuality.copyTrajectories = True


zeroStepFilter = cms.EDProducer("QualityFilter",
     TrackQuality = cms.string('highPurity'),
     recTracks = cms.InputTag("zeroStepTracksWithQuality:")
)

iterativeZeroTrackFiltering = cms.Sequence(zeroStepTracksWithQuality+zeroStepFilter)




# Final sequence
iterativeInitialStep = cms.Sequence(iterativeInitialSeeds
                                      +iterativeInitialTrackCandidates
                                      +iterativeInitialTracks
                                      +iterativeZeroTrackMerging
                                      +iterativeZeroTrackFiltering)


