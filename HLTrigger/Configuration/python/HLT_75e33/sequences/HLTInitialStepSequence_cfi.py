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
from ..modules.hltPixelSeedInputLST_cfi import *
from ..modules.hltSiPhase2RecHits_cfi import *
from ..modules.hltPhase2OTHitsInputLST_cfi import *
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
    +hltPixelSeedInputLST
    +hltSiPhase2RecHits # Probably need to move elsewhere in the final setup
    +hltPhase2OTHitsInputLST # Probably need to move elsewhere in the final setup
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

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceLST)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
(singleIterPatatrack & trackingLST).toReplaceWith(HLTInitialStepSequence, HLTInitialStepSequence.copyAndExclude([HLTHighPtTripletStepSeedingSequence,hltHighPtTripletStepSeedTracksLST]))

from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
(seedingLST & trackingLST).toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceLST.copyAndExclude([hltInitialStepTrackspLSTCLST,hltInitialStepTrackCutClassifierpLSTCLST,hltInitialStepTrackSelectionHighPuritypLSTCLST]))

