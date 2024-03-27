import FWCore.ParameterSet.Config as cms

from ..modules.initialStepSeeds_cfi import *
from ..modules.initialStepTrackCandidates_cfi import *
from ..modules.initialStepTrackCutClassifier_cfi import *
from ..modules.initialStepTracks_cfi import *
from ..modules.initialStepTrackSelectionHighPurity_cfi import *

initialStepSequence = cms.Sequence(initialStepSeeds+initialStepTrackCandidates+initialStepTracks+initialStepTrackCutClassifier+initialStepTrackSelectionHighPurity)
