import FWCore.ParameterSet.Config as cms

from ..sequences.HLTPhase2MuonPixelTracksFromL1TkSequence_cfi import *
from ..sequences.HLTIter0Phase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTIter2Phase2L3FromL1TkSequence_cfi import *

HLTPhase2L3MuonsIOSequence = cms.Sequence(
    HLTPhase2MuonPixelTracksFromL1TkSequence
    + HLTIter0Phase2L3FromL1TkSequence
    + HLTIter2Phase2L3FromL1TkSequence
)

from ..modules.hltPhase2MuonPixelSeedsFromPixelTracks_cfi import *
from ..modules.hltPhase2MuonIOTrackCandidates_cfi import *
from ..modules.hltPhase2MuonIOTracks_cfi import *
_HLTPhase2L3MuonsIOSequencePixelSelector = cms.Sequence(
    HLTPhase2MuonPixelTracksFromL1TkSequence
    + hltPhase2MuonPixelSeedsFromPixelTracks
    + hltPhase2MuonIOTrackCandidates
    + hltPhase2MuonIOTracks
)

from Configuration.ProcessModifiers.phase2MuonPixelTracksSelector_cff import phase2MuonPixelTracksSelector
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
(phase2MuonPixelTracksSelector | ngtScouting).toReplaceWith(
    HLTPhase2L3MuonsIOSequence,
    _HLTPhase2L3MuonsIOSequencePixelSelector
)

# Tracking imports
from ..sequences.HLTTrackingSequence_cfi import *
from ..modules.hltInputLST_cfi import *
from ..modules.hltInitialStepSeeds_cfi import *
from ..modules.hltInitialStepTrajectorySeedsLST_cfi import *
from ..modules.hltLST_cfi import *
from ..modules.hltSiPhase2RecHits_cfi import *
from ..sequences.HLTMkFitInputSequence_cfi import *
# Muon imports
from ..modules.hltPhase2MuonIOTrackSeeds_cfi import *
from ..modules.hltPhase2MuonIOTrackMkFitSeeds_cfi import *
from ..modules.hltPhase2MuonIOTrackCandidatesMkFit_cfi import *
from ..modules.hltPhase2MuonIOTrackSelectionHighPurityForest_cfi import *
_HLTPhase2L3MuonsIOSequenceSeedsSelector = cms.Sequence(
    # Seeding on the full tracker (pixel tracks and LST)
    HLTItLocalRecoSequence
    + HLTOtLocalRecoSequence
    + hltTrackerClusterCheck
    + HLTPhase2PixelTracksAndVerticesSequence
    + hltInitialStepSeeds
    + hltSiPhase2RecHits
    + hltInputLST
    + hltLST
    + hltInitialStepTrajectorySeedsLST
    # Selection of the muon seeds, then track building and fit
    + hltPhase2MuonIOTrackSeeds
    + HLTMkFitInputSequence
    + hltPhase2MuonIOTrackMkFitSeeds
    + hltPhase2MuonIOTrackCandidatesMkFit
    + hltPhase2MuonIOTrackCandidates
    + hltPhase2MuonIOTracks
    + hltPhase2MuonIOTrackSelectionHighPurityForest
)

from Configuration.ProcessModifiers.phase2MuonSeedsSelector_cff import phase2MuonSeedsSelector
phase2MuonSeedsSelector.toReplaceWith(
    HLTPhase2L3MuonsIOSequence,
    _HLTPhase2L3MuonsIOSequenceSeedsSelector
)
