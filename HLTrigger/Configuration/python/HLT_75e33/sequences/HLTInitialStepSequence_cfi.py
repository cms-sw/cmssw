import FWCore.ParameterSet.Config as cms

from ..modules.hltInitialStepSeeds_cfi import *
from ..modules.hltInitialStepTrackCandidates_cfi import *
from ..modules.hltInitialStepTrackCutClassifier_cfi import *
from ..modules.hltInitialStepTracks_cfi import *
from ..modules.hltInitialStepTrackSelectionHighPurity_cfi import *

HLTInitialStepSequence = cms.Sequence(hltInitialStepSeeds+hltInitialStepTrackCandidates+hltInitialStepTracks+hltInitialStepTrackCutClassifier+hltInitialStepTrackSelectionHighPurity)

from ..modules.hltInitialStepSeedTracksLST_cfi import *
from ..sequences.HLTHighPtTripletStepSeedingSequence_cfi import *
from ..modules.hltHighPtTripletStepSeedTracksLST_cfi import *
from ..modules.hltSiPhase2RecHits_cfi import *
from ..modules.hltInputLST_cfi import *
from ..modules.hltLST_cfi import *
from ..modules.hltInitialStepTrackspTTCLST_cfi import *
from ..modules.hltInitialStepTrackspLSTCLST_cfi import *
from ..modules.hltInitialStepTracksT5TCLST_cfi import *
from ..modules.hltInitialStepTrackCutClassifierpTTCLST_cfi import *
from ..modules.hltInitialStepTrackCutClassifierpLSTCLST_cfi import *
from ..modules.hltInitialStepTrackSelectionHighPuritypTTCLST_cfi import *
from ..modules.hltInitialStepTrackSelectionHighPuritypLSTCLST_cfi import *
_HLTInitialStepSequenceLST = cms.Sequence(
     hltInitialStepSeeds
    +hltInitialStepSeedTracksLST
    +HLTHighPtTripletStepSeedingSequence
    +hltHighPtTripletStepSeedTracksLST
    +hltSiPhase2RecHits # Probably need to move elsewhere in the final setup
    +hltInputLST
    +hltLST
    +hltInitialStepTrackCandidates
    +hltInitialStepTrackspTTCLST
    +hltInitialStepTrackspLSTCLST
    +hltInitialStepTracksT5TCLST
    +hltInitialStepTrackCutClassifierpTTCLST
    +hltInitialStepTrackCutClassifierpLSTCLST
    +hltInitialStepTrackSelectionHighPuritypTTCLST
    +hltInitialStepTrackSelectionHighPuritypLSTCLST
)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.seedingLST_cff import seedingLST

(~singleIterPatatrack & trackingLST & ~seedingLST).toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceLST)

(singleIterPatatrack & trackingLST & ~seedingLST).toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceLST.copyAndExclude([HLTHighPtTripletStepSeedingSequence,hltHighPtTripletStepSeedTracksLST]))

(~singleIterPatatrack & trackingLST & seedingLST).toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceLST.copyAndExclude([hltInitialStepTrackspLSTCLST,hltInitialStepTrackCutClassifierpLSTCLST,hltInitialStepTrackSelectionHighPuritypLSTCLST]))

from ..modules.hltInitialStepTrajectorySeedsLST_cfi import *
_HLTInitialStepSequenceSingleIterPatatrackLSTSeeding = cms.Sequence(
     hltInitialStepSeeds
    +hltInitialStepSeedTracksLST
    +hltSiPhase2RecHits # Probably need to move elsewhere in the final setup
    +hltInputLST
    +hltLST
    +hltInitialStepTrajectorySeedsLST
    +hltInitialStepTrackCandidates
    +hltInitialStepTracks
)

(singleIterPatatrack & trackingLST & seedingLST).toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceSingleIterPatatrackLSTSeeding)
