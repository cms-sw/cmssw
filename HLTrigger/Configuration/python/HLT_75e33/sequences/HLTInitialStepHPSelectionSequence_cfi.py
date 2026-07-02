import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTrackCutClassifier_cfi import *
from ..modules.hltInitialStepTrackSelectionHighPurity_cfi import *

HLTInitialStepHPSelectionSequence = cms.Sequence(
    hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)
