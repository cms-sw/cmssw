import FWCore.ParameterSet.Config as cms

from ..modules.hltRpcRecHits_cfi import *
from ..sequences.HLTCsclocalrecoSequence_cfi import *
from ..sequences.HLTDtlocalrecoSequence_cfi import *
from ..sequences.HLTGemLocalRecoSequence_cfi import *

HLTMuonlocalrecoSequence = cms.Sequence(HLTCsclocalrecoSequence+HLTDtlocalrecoSequence+HLTGemLocalRecoSequence+hltRpcRecHits)
