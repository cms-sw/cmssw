import FWCore.ParameterSet.Config as cms

from ..modules.hltHighPtTripletStepClusters_cfi import *
from ..modules.hltHighPtTripletStepHitDoublets_cfi import *
from ..modules.hltHighPtTripletStepHitTriplets_cfi import *
from ..modules.hltHighPtTripletStepSeedLayers_cfi import *
from ..modules.hltHighPtTripletStepSeeds_cfi import *

HLTHighPtTripletStepSeedingSequence = cms.Sequence(hltHighPtTripletStepClusters+hltHighPtTripletStepSeedLayers+hltHighPtTripletStepHitDoublets+hltHighPtTripletStepHitTriplets+hltHighPtTripletStepSeeds)
