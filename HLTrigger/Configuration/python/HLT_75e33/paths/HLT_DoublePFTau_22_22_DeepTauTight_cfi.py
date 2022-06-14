import FWCore.ParameterSet.Config as cms

from ..modules.hltPreDoublePFTau2222DeepTauTight_cfi import *
from ..modules.hltSelectedHpsPFTaus8HitsMaxDeltaZWithOfflineVertices_cfi import *
from ..modules.hltHpsDoublePFTau22_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..tasks.HLTTauTask_cff import *
from ..sequences.HLTEndSequence_cfi import *

HLT_DoublePFTau_22_22_DeepTauTight = cms.Path(
    HLTBeginSequence +
    hltPreDoublePFTau2222DeepTauTight + 
    hltSelectedHpsPFTaus8HitsMaxDeltaZWithOfflineVertices + 
    hltHpsDoublePFTau22 + 
    HLTEndSequence,
    HLTTauTask

)
