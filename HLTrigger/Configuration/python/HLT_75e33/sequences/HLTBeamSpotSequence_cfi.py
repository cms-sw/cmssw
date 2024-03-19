import FWCore.ParameterSet.Config as cms

from ..modules.hltOnlineBeamSpot_cfi import *

HLTBeamSpotSequence = cms.Sequence(hltOnlineBeamSpot)
