import FWCore.ParameterSet.Config as cms

from ..tasks.HLTTauTask_cff import *

HLTDoublePFTau2222DeepTauTightSequence = cms.Sequence(
                                       HLTTauTask+
                                       hltSelectedHpsPFTaus8HitsMaxDeltaZWithOfflineVertices
)