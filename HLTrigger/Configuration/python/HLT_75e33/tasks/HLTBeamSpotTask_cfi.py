import FWCore.ParameterSet.Config as cms

from ..modules.hltOnlineBeamSpot_cfi import *

HLTBeamSpotTask = cms.Task(
    hltOnlineBeamSpot
)
