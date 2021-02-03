import FWCore.ParameterSet.Config as cms

from ..modules.muonSeededSeedsInOut_cfi import *
from ..modules.muonSeededTrackCandidatesInOut_cfi import *
from ..modules.muonSeededTracksInOut_cfi import *

muonSeededStepCoreInOutTask = cms.Task(muonSeededSeedsInOut, muonSeededTrackCandidatesInOut, muonSeededTracksInOut)
