import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTrackCutClassifier_cfi import *
from ..modules.hltInitialStepTrackSelectionHighPurity_cfi import *
from ..modules.hltInitialStepTrackFeatureExtractor_cfi import *
from ..modules.hltInitialStepTrackTorchClassifier_cfi import *
from ..modules.hltInitialStepTrackTorchClassifierOutput_cfi import *

HLTInitialStepTorchClassifierHPSelectionSequence = cms.Sequence(
    hltInitialStepTrackFeatureExtractor
    +hltInitialStepTrackTorchClassifier
    +hltInitialStepTrackTorchClassifierOutput 
    +hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)
