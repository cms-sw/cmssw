import FWCore.ParameterSet.Config as cms

from ..modules.highPtTripletStepTrackCandidates_cfi import *
from ..modules.highPtTripletStepTrackCutClassifier_cfi import *
from ..modules.highPtTripletStepTracks_cfi import *
from ..modules.highPtTripletStepTrackSelectionHighPurity_cfi import *
from ..tasks.highPtTripletStepSeedingTask_cfi import *

highPtTripletStepTask = cms.Task(highPtTripletStepSeedingTask, highPtTripletStepTrackCandidates, highPtTripletStepTrackCutClassifier, highPtTripletStepTrackSelectionHighPurity, highPtTripletStepTracks)
