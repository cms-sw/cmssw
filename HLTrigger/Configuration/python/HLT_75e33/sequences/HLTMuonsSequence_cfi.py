import FWCore.ParameterSet.Config as cms

from ..sequences.HLTIter0Phase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTIter2Phase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTL2MuonsFromL1TkSequence_cfi import *
from ..sequences.HLTPhase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTPhase2L3MuonsSequence_cfi import *
from ..sequences.HLTPhase2L3OISequence_cfi import *
from ..modules.hltPhase2L3MuonFilter_cfi import *

# The default HLT Muons sequence (Inside-Out first)
HLTMuonsSequence = cms.Sequence(
    HLTL2MuonsFromL1TkSequence
    + HLTPhase2L3FromL1TkSequence
    + HLTIter0Phase2L3FromL1TkSequence
    + HLTIter2Phase2L3FromL1TkSequence
    + hltPhase2L3MuonFilter
    + HLTPhase2L3OISequence
    + HLTPhase2L3MuonsSequence
)

# Outside-In first HLT Muons sequence
_HLTMuonsSequenceOIFirst = cms.Sequence(
    HLTL2MuonsFromL1TkSequence
    + HLTPhase2L3OISequence
    + hltPhase2L3MuonFilter
    + HLTPhase2L3FromL1TkSequence
    + HLTIter0Phase2L3FromL1TkSequence
    + HLTIter2Phase2L3FromL1TkSequence
    + HLTPhase2L3MuonsSequence
)

from Configuration.ProcessModifiers.phase2L3MuonsOIFirst_cff import phase2L3MuonsOIFirst
phase2L3MuonsOIFirst.toReplaceWith(HLTMuonsSequence, _HLTMuonsSequenceOIFirst)

