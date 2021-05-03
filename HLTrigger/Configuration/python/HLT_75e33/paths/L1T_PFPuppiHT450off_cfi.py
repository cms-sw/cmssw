import FWCore.ParameterSet.Config as cms

from ..modules.l1tPFPuppiHT_cfi import *
from ..modules.l1tPFPuppiHT450off_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

L1T_PFPuppiHT450off = cms.Path(HLTBeginSequence+l1tPFPuppiHT+l1tPFPuppiHT450off+HLTEndSequence)
