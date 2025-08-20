import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2L3GlbMuon_cfi import *
from ..modules.hltPhase2L3MuonMerged_cfi import *
from ..modules.hltPhase2L3Muons_cfi import *
from ..modules.hltPhase2L3MuonsNoID_cfi import *
from ..modules.hltPhase2L3MuonCandidates_cfi import *

HLTPhase2L3MuonsSequence = cms.Sequence(
    hltPhase2L3MuonMerged
    + hltPhase2L3GlbMuon
    + hltPhase2L3MuonsNoID
    + hltPhase2L3Muons
    + hltPhase2L3MuonCandidates
)
