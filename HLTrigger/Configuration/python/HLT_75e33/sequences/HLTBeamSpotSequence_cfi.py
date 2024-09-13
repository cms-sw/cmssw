import FWCore.ParameterSet.Config as cms

from ..modules.hltOnlineBeamSpot_cfi import *

HLTBeamSpotSequence = cms.Sequence(hltOnlineBeamSpot)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
from ..modules.hltPhase2OnlineBeamSpotDevice_cfi import hltPhase2OnlineBeamSpotDevice
_HLTBeamSpotSequence = cms.Sequence(
     hltOnlineBeamSpot
    +hltPhase2OnlineBeamSpotDevice
)
alpaka.toReplaceWith(HLTBeamSpotSequence, _HLTBeamSpotSequence)
