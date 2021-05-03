import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *

HLTDoubleEle25CaloIdLPMS2UnseededTask = cms.Task(hltEgammaCandidatesUnseeded, hltEgammaClusterShapeUnseeded, hltEgammaHGCALIDVarsUnseeded, hltEgammaHoverEUnseeded)
