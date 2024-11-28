import FWCore.ParameterSet.Config as cms

from ..sequences.HLTIter0Phase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTIter2Phase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTL2MuonsFromL1TkSequence_cfi import *
from ..sequences.HLTPhase2L3FromL1TkSequence_cfi import *
from ..sequences.HLTPhase2L3MuonsSequence_cfi import *
from ..sequences.HLTPhase2L3OISequence_cfi import *

HLTMuonsSequence = cms.Sequence(
    HLTL2MuonsFromL1TkSequence
    + HLTPhase2L3OISequence
    + HLTPhase2L3FromL1TkSequence
    + HLTIter0Phase2L3FromL1TkSequence
    + HLTIter2Phase2L3FromL1TkSequence
    + HLTPhase2L3MuonsSequence
)

from ..modules.hltPhase2L3MuonFilter_cfi import *

# The IO first HLT Muons sequence
Phase2HLTMuonsSequenceIOFirst = cms.Sequence(
    HLTL2MuonsFromL1TkSequence
    + HLTPhase2L3FromL1TkSequence
    + HLTIter0Phase2L3FromL1TkSequence
    + HLTIter2Phase2L3FromL1TkSequence
    + hltPhase2L3MuonFilter
    + HLTPhase2L3OISequence
    + HLTPhase2L3MuonsSequence
)
# The OI first HLT Muons sequence
Phase2HLTMuonsSequenceOIFirst = cms.Sequence(
    HLTL2MuonsFromL1TkSequence
    + HLTPhase2L3OISequence
    + hltPhase2L3MuonFilter
    + HLTPhase2L3FromL1TkSequence
    + HLTIter0Phase2L3FromL1TkSequence
    + HLTIter2Phase2L3FromL1TkSequence
    + HLTPhase2L3MuonsSequence
)

from Configuration.ProcessModifiers.phase2L2AndL3Muons_cff import phase2L2AndL3Muons

phase2L2AndL3Muons.toReplaceWith(HLTMuonsSequence, Phase2HLTMuonsSequenceIOFirst)

from Configuration.ProcessModifiers.phase2L3MuonsOIFirst_cff import phase2L3MuonsOIFirst

(phase2L2AndL3Muons & phase2L3MuonsOIFirst).toReplaceWith(
    HLTMuonsSequence, Phase2HLTMuonsSequenceOIFirst
)
