import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2L3OIMuCtfWithMaterialTracks_cfi import *
from ..modules.hltPhase2L3OIMuonTrackCutClassifier_cfi import *
from ..modules.hltPhase2L3OIMuonTrackSelectionHighPurity_cfi import *
from ..modules.hltPhase2L3OISeedsFromL2Muons_cfi import *
from ..modules.hltPhase2L3OITrackCandidates_cfi import *

HLTPhase2L3OISequence = cms.Sequence(
    hltPhase2L3OISeedsFromL2Muons
    + hltPhase2L3OITrackCandidates
    + hltPhase2L3OIMuCtfWithMaterialTracks
    + hltPhase2L3OIMuonTrackCutClassifier
    + hltPhase2L3OIMuonTrackSelectionHighPurity
)
