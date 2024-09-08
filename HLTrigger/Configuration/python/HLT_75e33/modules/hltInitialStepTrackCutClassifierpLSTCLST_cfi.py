import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepTrackCutClassifier_cfi import hltInitialStepTrackCutClassifier as _hltInitialStepTrackCutClassifier
hltInitialStepTrackCutClassifierpLSTCLST = _hltInitialStepTrackCutClassifier.clone( src = "hltInitialStepTrackspLSTCLST" )

