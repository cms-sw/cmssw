import FWCore.ParameterSet.Config as cms

from ..modules.hltPreDoubleEle25CaloIdLPMS2Unseeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTDoubleEle25CaloIdLPMS2UnseededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_DoubleEle25_CaloIdL_PMS2_Unseeded = cms.Path(HLTBeginSequence+hltPreDoubleEle25CaloIdLPMS2Unseeded+HLTDoubleEle25CaloIdLPMS2UnseededSequence+HLTEndSequence)
