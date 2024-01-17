import FWCore.ParameterSet.Config as cms

from ..modules.highPtTripletStepClusters_cfi import *
from ..modules.highPtTripletStepHitDoublets_cfi import *
from ..modules.highPtTripletStepHitTriplets_cfi import *
from ..modules.highPtTripletStepSeedLayers_cfi import *
from ..modules.highPtTripletStepSeeds_cfi import *

highPtTripletStepSeedingSequence = cms.Sequence(highPtTripletStepClusters+highPtTripletStepSeedLayers+highPtTripletStepHitDoublets+highPtTripletStepHitTriplets+highPtTripletStepSeeds)
