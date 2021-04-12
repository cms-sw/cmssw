import FWCore.ParameterSet.Config as cms

from ..tasks.highPtTripletStepSeedingTask_cfi import *

highPtTripletStepSeeding = cms.Sequence(highPtTripletStepSeedingTask)
