import FWCore.ParameterSet.Config as cms

from ..modules.highPtTripletStepClusters_cfi import *
from ..modules.highPtTripletStepHitDoublets_cfi import *
from ..modules.highPtTripletStepHitTriplets_cfi import *
from ..modules.highPtTripletStepSeedLayers_cfi import *
from ..modules.highPtTripletStepSeeds_cfi import *
from ..modules.hltPhase2PixelTracksAndHighPtStepTrackingRegions_cfi import *

highPtTripletStepSeedingTask = cms.Task(
    highPtTripletStepClusters,
    highPtTripletStepHitDoublets,
    highPtTripletStepHitTriplets,
    highPtTripletStepSeedLayers,
    highPtTripletStepSeeds,
    hltPhase2PixelTracksAndHighPtStepTrackingRegions
)
