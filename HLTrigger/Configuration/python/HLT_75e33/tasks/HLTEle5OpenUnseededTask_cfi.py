import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoUnseeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoV6Unseeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoUnseeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *

HLTEle5OpenUnseededTask = cms.Task(
    hltEgammaCandidatesUnseeded,
    hltEgammaClusterShapeUnseeded,
    hltEgammaEcalPFClusterIsoUnseeded,
    hltEgammaEleGsfTrackIsoUnseeded,
    hltEgammaEleGsfTrackIsoV6Unseeded,
    hltEgammaEleL1TrkIsoUnseeded,
    hltEgammaHGCALIDVarsUnseeded,
    hltEgammaHGCalLayerClusterIsoUnseeded,
    hltEgammaHcalPFClusterIsoUnseeded,
    hltEgammaHoverEUnseeded
)
