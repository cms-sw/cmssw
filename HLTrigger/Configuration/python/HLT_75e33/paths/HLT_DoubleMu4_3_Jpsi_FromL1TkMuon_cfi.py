import FWCore.ParameterSet.Config as cms

from ..modules.hltDoubleMu43JPsiL3Filtered_cfi import *
from ..modules.hltDisplacedmumuVtxProducerDoubleMu43Jpsi_cfi import *
from ..modules.hltDisplacedmumuFilterDoubleMu43Jpsi_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTDoubleMuonLowMassFromL1TkMuonSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_DoubleMu4_3_Jpsi_FromL1TkMuon = cms.Path(
    HLTBeginSequence
    + HLTDoubleMuonLowMassFromL1TkMuonSequence
    + hltDoubleMu43JPsiL3Filtered
    + hltDisplacedmumuVtxProducerDoubleMu43Jpsi
    + hltDisplacedmumuFilterDoubleMu43Jpsi
    + HLTEndSequence
)
