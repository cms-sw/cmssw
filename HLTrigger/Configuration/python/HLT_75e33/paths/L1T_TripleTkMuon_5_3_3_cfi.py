import FWCore.ParameterSet.Config as cms

from ..modules.hltL1SingleMuFiltered5_cfi import *
from ..modules.hltL1TripleMuFiltered3_cfi import *
from ..modules.hltTripleMuon3DR0_cfi import *
from ..modules.hltTripleMuon3DZ1p0_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

L1T_TripleTkMuon_5_3_3 = cms.Path(
    HLTBeginSequence +
    hltL1TripleMuFiltered3 +
    hltL1SingleMuFiltered5 +
    hltTripleMuon3DZ1p0 +
    hltTripleMuon3DR0 +
    HLTEndSequence
)
