import FWCore.ParameterSet.Config as cms

from ..modules.hltDoubleMu4BsL3Filtered_cfi import *
from ..modules.hltDisplacedmumuVtxProducerDoubleMu4Bs_cfi import *
from ..modules.hltDisplacedmumuFilterDoubleMu4Bs_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTDoubleMuonLowMassFromL1TkMuonSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_DoubleMu4_3_Bs_FromL1TkMuon = cms.Path(
    HLTBeginSequence
    + HLTDoubleMuonLowMassFromL1TkMuonSequence
    + hltDoubleMu4BsL3Filtered
    + hltDisplacedmumuVtxProducerDoubleMu4Bs
    + hltDisplacedmumuFilterDoubleMu4Bs
    + HLTEndSequence
)
