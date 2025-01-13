import FWCore.ParameterSet.Config as cms

from ..modules.hltIter0Phase2L3FromL1TkMuonCkfTrackCandidates_cfi import *
from ..modules.hltIter0Phase2L3FromL1TkMuonCtfWithMaterialTracks_cfi import *
from ..modules.hltIter0Phase2L3FromL1TkMuonPixelSeedsFromPixelTracks_cfi import *
from ..modules.hltIter0Phase2L3FromL1TkMuonTrackCutClassifier_cfi import *
from ..modules.hltIter0Phase2L3FromL1TkMuonTrackSelectionHighPurity_cfi import *

HLTIter0Phase2L3FromL1TkSequence = cms.Sequence(
    hltIter0Phase2L3FromL1TkMuonPixelSeedsFromPixelTracks
    + hltIter0Phase2L3FromL1TkMuonCkfTrackCandidates
    + hltIter0Phase2L3FromL1TkMuonCtfWithMaterialTracks
    + hltIter0Phase2L3FromL1TkMuonTrackCutClassifier
    + hltIter0Phase2L3FromL1TkMuonTrackSelectionHighPurity
)
