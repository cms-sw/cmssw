import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *

HLTDiphoton3023IsoCaloIdUnseededTask = cms.Task(
    hltEgammaCandidatesUnseeded,
    hltEgammaClusterShapeUnseeded,
    hltEgammaEcalPFClusterIsoUnseeded,
    hltEgammaHGCALIDVarsUnseeded,
    hltEgammaHGCalLayerClusterIsoUnseeded,
    hltEgammaHcalPFClusterIsoUnseeded,
    hltEgammaHoverEUnseeded
)
