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
