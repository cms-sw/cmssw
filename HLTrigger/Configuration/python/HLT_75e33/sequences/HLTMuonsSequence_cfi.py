import FWCore.ParameterSet.Config as cms

from ..sequences.HLTL2MuonsFromL1TkSequence_cfi import *
from ..sequences.HLTPhase2L3MuonsIOSequence_cfi import *
from ..sequences.HLTPhase2L3MuonsOISequence_cfi import *
from ..sequences.HLTPhase2MuonIdSequence_cfi import *
from ..modules.hltPhase2L3MuonFilter_cfi import *

# The default HLT Muons sequence (Inside-Out first)
HLTMuonsSequence = cms.Sequence(
    HLTL2MuonsFromL1TkSequence
    + HLTPhase2L3MuonsIOSequence
    + hltPhase2L3MuonFilter
    + HLTPhase2L3MuonsOISequence
    + HLTPhase2MuonIdSequence
)

# Outside-In first HLT Muons sequence
_HLTMuonsSequenceOIFirst = cms.Sequence(
    HLTL2MuonsFromL1TkSequence
    + HLTPhase2L3MuonsOISequence
    + hltPhase2L3MuonFilter
    + HLTPhase2L3MuonsIOSequence
    + HLTPhase2MuonIdSequence
)

from Configuration.ProcessModifiers.phase2L3MuonsOIFirst_cff import phase2L3MuonsOIFirst
phase2L3MuonsOIFirst.toReplaceWith(HLTMuonsSequence, _HLTMuonsSequenceOIFirst)

