import FWCore.ParameterSet.Config as cms

from ..modules.l1tPFPuppiMET220off_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

L1T_PFPuppiMET220off = cms.Path(
    HLTBeginSequence +
    l1tPFPuppiMET220off +
    HLTEndSequence
)
