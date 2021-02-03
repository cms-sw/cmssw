import FWCore.ParameterSet.Config as cms

from ..modules.highPtTripletStepTrackCandidates_cfi import *
from ..modules.highPtTripletStepTrackCutClassifier_cfi import *
from ..modules.highPtTripletStepTracks_cfi import *
from ..modules.highPtTripletStepTrackSelectionHighPurity_cfi import *
from ..sequences.highPtTripletStepSeeding_cfi import *

highPtTripletStepSequence = cms.Sequence(highPtTripletStepSeeding+highPtTripletStepTrackCandidates+highPtTripletStepTracks+highPtTripletStepTrackCutClassifier+highPtTripletStepTrackSelectionHighPurity)
