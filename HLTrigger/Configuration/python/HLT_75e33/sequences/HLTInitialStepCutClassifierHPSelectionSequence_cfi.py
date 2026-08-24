import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTrackCutClassifier_cfi import *
from ..modules.hltInitialStepTrackSelectionHighPurity_cfi import *

HLTInitialStepCutClassifierHPSelectionSequence = cms.Sequence(
    hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)
