import FWCore.ParameterSet.Config as cms

from ..modules.hltOnlineBeamSpot_cfi import *
from ..modules.hltPhase2OnlineBeamSpotDevice_cfi import hltPhase2OnlineBeamSpotDevice

HLTBeamSpotSequence = cms.Sequence(
     hltOnlineBeamSpot
    +hltPhase2OnlineBeamSpotDevice
)
