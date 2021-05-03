import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *

HLTPhoton187UnseededTask = cms.Task(hltEgammaCandidatesUnseeded, hltEgammaHGCALIDVarsUnseeded, hltEgammaHoverEUnseeded)
