import FWCore.ParameterSet.Config as cms

from ..modules.hltScalersRawToDigi_cfi import *
from ..modules.offlineBeamSpot_cfi import *

HLTBeamSpot = cms.Sequence(hltScalersRawToDigi+offlineBeamSpot)
