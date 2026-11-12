import FWCore.ParameterSet.Config as cms

from ..modules.hltDoubleMu43LowMassL3Filtered_cfi import *
from ..modules.hltDisplacedmumuVtxProducerDoubleMu43LowMass_cfi import *
from ..modules.hltDisplacedmumuFilterDoubleMu43LowMass_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTDoubleMuonLowMassFromL1TkMuonSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_DoubleMu4_3_LowMass_FromL1TkMuon = cms.Path(
    HLTBeginSequence
    + HLTDoubleMuonLowMassFromL1TkMuonSequence
    + hltDoubleMu43LowMassL3Filtered
    + hltDisplacedmumuVtxProducerDoubleMu43LowMass
    + hltDisplacedmumuFilterDoubleMu43LowMass
    + HLTEndSequence
)
