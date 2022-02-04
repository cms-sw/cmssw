import FWCore.ParameterSet.Config as cms

from ..modules.hltOnlineBeamSpot_cfi import *
from ..modules.hltScalersRawToDigi_cfi import *
from ..modules.offlineBeamSpot_cfi import *

HLTBeamSpotTask = cms.Task(
    hltOnlineBeamSpot,
    hltScalersRawToDigi,
    offlineBeamSpot
)
