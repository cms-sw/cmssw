import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

from ..modules.hltInputLST_cfi import *
from ..modules.hltInitialStepMkFitSeeds_cfi import *
from ..modules.hltInitialStepSeedTracksLST_cfi import *
from ..modules.hltInitialStepSeeds_cfi import *
from ..modules.hltInitialStepTrackCandidates_cfi import *
from ..modules.hltInitialStepTrackCandidatesMkFit_cfi import *
from ..modules.hltInitialStepTrackCutClassifier_cfi import *
from ..modules.hltInitialStepTrackSelectionHighPurity_cfi import *
from ..modules.hltInitialStepTracks_cfi import *
from ..modules.hltInitialStepTrajectorySeedsLST_cfi import *
from ..modules.hltInitialStepTrajectorySeedsLSTTracks_cfi import *
from ..modules.hltLST_cfi import *
from ..modules.hltSiPhase2RecHits_cfi import *
from ..sequences.HLTMkFitInputSequence_cfi import *

HLTInitialStepSequence = cms.Sequence(
     hltInitialStepSeeds
    +hltInitialStepSeedTracksLST
    +hltSiPhase2RecHits
    +hltInputLST
    +hltLST
    +hltInitialStepTrajectorySeedsLST
    +HLTMkFitInputSequence
    +hltInitialStepMkFitSeeds
    +hltInitialStepTrackCandidatesMkFit
    +hltInitialStepTrackCandidates
    +hltInitialStepTracks
    +hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)


# Empty sequence as a placeholder to be filled when alpakaValidationHLT is active
HLTInitialStepSequenceSerialSync = cms.Sequence()

hltInitialStepSeedsSerialSync = hltInitialStepSeeds.clone(
    InputCollection = "hltPhase2PixelTracksSerialSync"
)
hltInitialStepSeedTracksLSTSerialSync = hltInitialStepSeedTracksLST.clone(
    src = "hltInitialStepSeedsSerialSync"
)
hltInputLSTSerialSync = makeSerialClone(hltInputLST)
hltLSTSerialSync = makeSerialClone(hltLST,
    lstInput = "hltInputLSTSerialSync"
)
hltInitialStepTrajectorySeedsLSTSerialSync = hltInitialStepTrajectorySeedsLST.clone(
    lstOutput = "hltLSTSerialSync",
    lstInput = "hltInputLSTSerialSync",
    lstPixelSeeds = "hltInputLSTSerialSync"
)
hltInitialStepTrajectorySeedsLSTTracksSerialSync = hltInitialStepTrajectorySeedsLSTTracks.clone(
    src = "hltInitialStepTrajectorySeedsLSTSerialSync"
)
# Sequence for CPU vs. GPU validation, to be kept in sync with nominal one:
# A copy of the nominal sequence, dropping modules not used
# and adding a module needed for CPU vs. GPU comparisons
from Configuration.ProcessModifiers.alpakaValidationHLT_cff import alpakaValidationHLT
alpakaValidationHLT.toReplaceWith(HLTInitialStepSequenceSerialSync, cms.Sequence(
     hltInitialStepSeedsSerialSync
    +hltInitialStepSeedTracksLSTSerialSync
    +hltSiPhase2RecHits
    +hltInputLSTSerialSync
    +hltLSTSerialSync
    +hltInitialStepTrajectorySeedsLSTSerialSync
    +hltInitialStepTrajectorySeedsLSTTracksSerialSync
))

# A copy the nominal sequence, adding a module needed for CPU vs. GPU comparisons
_HLTHeterogeneousInitialStepSequence = HLTInitialStepSequence.copy()
_HLTHeterogeneousInitialStepSequence += hltInitialStepTrajectorySeedsLSTTracks
alpakaValidationHLT.toReplaceWith(HLTInitialStepSequence, _HLTHeterogeneousInitialStepSequence)


from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
hltPhase2LegacyTracking.toReplaceWith(HLTInitialStepSequence,
    HLTInitialStepSequence.copyAndExclude([
        hltInitialStepSeedTracksLST,
        hltSiPhase2RecHits,
        hltInputLST,
        hltLST,
        hltInitialStepTrajectorySeedsLST,
        HLTMkFitInputSequence,
        hltInitialStepMkFitSeeds,
        hltInitialStepTrackCandidatesMkFit
    ])
)


_HLTInitialStepSequenceLST = cms.Sequence(
    hltInitialStepSeeds
    +hltInitialStepSeedTracksLST
    +hltSiPhase2RecHits # Probably need to move elsewhere in the final setup
    +hltInputLST
    +hltLST
    +hltInitialStepTrackCandidates
    +hltInitialStepTracks
    +hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceLST)


from ..modules.hltInitialStepTracksT4T5TCLST_cfi import *
_HLTInitialStepSequenceNGTScouting = cms.Sequence(
    hltInitialStepSeeds
    +hltInitialStepSeedTracksLST
    +hltSiPhase2RecHits
    +hltInputLST
    +hltLST
    +hltInitialStepTrackCandidates
    +hltInitialStepTracksT4T5TCLST
)

from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
ngtScouting.toReplaceWith(HLTInitialStepSequence,_HLTInitialStepSequenceNGTScouting)


from ..modules.hltInitialStepTrackCandidatesMkFitFit_cfi import *
_HLTInitialStepSequenceMkFitFit = cms.Sequence(
    hltInitialStepSeeds
    +hltInitialStepSeedTracksLST
    +hltSiPhase2RecHits
    +hltInputLST
    +hltLST
    +hltInitialStepTrajectorySeedsLST
    +HLTMkFitInputSequence
    +hltInitialStepMkFitSeeds
    +hltInitialStepTrackCandidatesMkFit
    +hltInitialStepTrackCandidatesMkFitFit
    +hltInitialStepTracks
    +hltInitialStepTrackCutClassifier
    +hltInitialStepTrackSelectionHighPurity
)

from Configuration.ProcessModifiers.trackingMkFitFit_cff import trackingMkFitFit
trackingMkFitFit.toReplaceWith(HLTInitialStepSequence, _HLTInitialStepSequenceMkFitFit)
