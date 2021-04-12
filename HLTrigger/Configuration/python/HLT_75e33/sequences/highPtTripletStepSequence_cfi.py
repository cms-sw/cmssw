import FWCore.ParameterSet.Config as cms

from ..tasks.highPtTripletStepTask_cfi import *

highPtTripletStepSequence = cms.Sequence(highPtTripletStepTask)
