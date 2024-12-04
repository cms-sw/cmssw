import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTrackCutClassifier_cfi import hltInitialStepTrackCutClassifier as _hltInitialStepTrackCutClassifier
hltInitialStepTrackCutClassifierpTTCLST = _hltInitialStepTrackCutClassifier.clone( src = "hltInitialStepTrackspTTCLST" )
