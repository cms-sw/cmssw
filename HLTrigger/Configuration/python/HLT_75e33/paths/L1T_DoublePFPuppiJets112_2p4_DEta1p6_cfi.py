import FWCore.ParameterSet.Config as cms

from ..modules.l1tDoublePFPuppiJet112offMaxEta2p4_cfi import *
from ..modules.l1tDoublePFPuppiJets112offMaxDeta1p6_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

L1T_DoublePFPuppiJets112_2p4_DEta1p6 = cms.Path(
    HLTBeginSequence +
    l1tDoublePFPuppiJet112offMaxEta2p4 +
    l1tDoublePFPuppiJets112offMaxDeta1p6 +
    HLTEndSequence
)
