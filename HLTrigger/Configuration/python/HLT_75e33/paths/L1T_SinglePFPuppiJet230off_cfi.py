import FWCore.ParameterSet.Config as cms

from ..modules.l1tSinglePFPuppiJet230off_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

L1T_SinglePFPuppiJet230off = cms.Path(
    HLTBeginSequence +
    l1tSinglePFPuppiJet230off +
    HLTEndSequence
)
