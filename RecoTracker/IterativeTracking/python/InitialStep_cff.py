import FWCore.ParameterSet.Config as cms

### STEP 0 ###

# hit building
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *

# seeding
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
initialStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.6,
    originRadius = 0.03,
    nSigmaZ = 4.0
    )
    )
    )
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'

# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
initialStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'initialStepTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    minimumNumberOfHits = 3,
    maxLostHits = 0,
    minPt = 0.4
    )
    )

import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
initialStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'initialStepTrajectoryBuilder',
    trajectoryFilterName = 'initialStepTrajectoryFilter'
    )

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
initialStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('initialStepSeeds'),
    TrajectoryBuilder = 'initialStepTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# fitting
import RecoTracker.TrackProducer.TrackProducer_cfi
initialStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'initialStepTrackCandidates',
    AlgorithmName = cms.string('iter0'),
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
InitialStep = cms.Sequence(initialStepSeeds*
                           initialStepTrackCandidates*
                           initialStepTracks*
                           initialStepSelector)
