import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *

HLTPhoton108EBTightIDTightIsoUnseededTask = cms.Task(
    hltEgammaCandidatesUnseeded,
    hltEgammaClusterShapeUnseeded,
    hltEgammaEcalPFClusterIsoUnseeded,
    hltEgammaHcalPFClusterIsoUnseeded,
    hltEgammaHoverEUnseeded
)
